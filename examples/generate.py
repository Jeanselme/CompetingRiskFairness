import numpy as np
import pandas as pd
from scipy.stats import gompertz
from sklearn.datasets import make_blobs


def _build_shapes(dim):
    """
    Build the cause/censoring/scale shape functions for a given dim.
    dim must be even (it is split into two equal halves) and >= 2.
    """
    if dim < 2 or dim % 2 != 0:
        raise ValueError("dim must be even and >= 2")
    half = dim // 2
    return {
        -1: lambda p, x: np.abs((p * x)[:, half:]).sum(1) / dim,                            # shared scale (w_s)
         0: lambda p, x: ((p * x)[:, half:] ** 2).sum(1) / dim,                              # censoring shape
         1: lambda p, x: np.abs(((p * x)[:, half:] ** 2) + ((p * x)[:, :half])).sum(1) / dim, # cause 1 shape
         2: lambda p, x: np.abs(((p * x)[:, :half] ** 2) + ((p * x)[:, half:])).sum(1) / dim, # cause 2 shape
         3: lambda p, x: np.abs((p * x)[:, :half] + (p * x)[:, half:]).sum(1) / dim,         # cause 3 shape (linear)
    }


def generate(random_seed=42, size=30000, causes=(1, 2), dim=10):
    """
    Generate two clusters with different survival profiles.
    Concatenate clusters and covariates for X.
    Model len(causes) competing outcomes.

    causes: list of competing risk labels to simulate. To add a 4th cause, add
            it here and define a matching entry in _build_shapes().
    dim: number of covariates that actually drive the hazards (even, >= 2).
    dim: total number of covariates returned in X (must be >= dim).
         Columns beyond dim are independent standard normal noise that
         does NOT enter any hazard, so increasing dim only adds nuisance
         features without weakening the separation between causes. They are
         drawn last so the cluster, hazards, durations, and events are
         identical across any dim >= dim for the same random_seed.

    causes and dim only need to be set here: compute_cif() and
    conditional_probability() read them back out of the returned `betas`,
    so there is no separate copy of this config to keep in sync elsewhere.

    return x, t, e, betas, z
    """
    causes = list(causes)
    if dim < dim:
        raise ValueError(f"dim must be >= dim ({dim})")
    shapes = _build_shapes(dim)

    np.random.seed(random_seed)

    # Data: two blobs for the first two covariates, the rest standard normal
    x, z = make_blobs(n_samples=size, n_features=2, centers=([-1.5, -1.5], [1.5, 1.5]))
    x_signal = np.column_stack([x] + [np.random.normal(size=size) for _ in range(dim - 2)])

    # Group-specific parameters for the shared scale (-1) and each cause
    parameters = {event: np.array([np.random.normal(size=dim) for _ in np.unique(z)])
                  for event in [-1] + causes}

    # Cause-specific hazards and the shared Gompertz scale, computed from the signal columns only
    cause_hazards = {r: shapes[r](parameters[r][z], x_signal) for r in causes}
    total_hazard = sum(cause_hazards.values())
    sc = shapes[-1](parameters[-1][z], x_signal)

    # gompertz.rvs(c, loc, scale) implies hazard h(t) = (c/scale)*exp(t/scale).
    # To exactly reproduce lambda_r(t) = w_r * exp(w_s * t) summed over causes,
    # c must be total_hazard / w_s and scale must be 1 / w_s (not total_hazard, w_s directly).
    outcomes = gompertz.rvs(total_hazard / sc, scale=1 / sc)

    # Draw the event type from a categorical distribution over the relative hazards.
    # This generalizes the original two-cause Bernoulli draw to any number of causes.
    probs = np.stack([cause_hazards[r] for r in causes], axis=1)
    probs = probs / probs.sum(1, keepdims=True)
    cum_probs = np.cumsum(probs, axis=1)
    u = np.random.uniform(size=size)
    event_idx = (u[:, None] > cum_probs).sum(1)
    events = np.array(causes)[event_idx]

    # Censoring time: fixed, group-independent parameters (not random across groups)
    censoring_beta = np.random.normal(size=dim)
    censoring = gompertz.rvs(shapes[0](censoring_beta, x_signal))

    is_observed = censoring > outcomes
    events = is_observed * events            # 0 marks censoring
    outcomes = np.where(is_observed, outcomes, censoring)

    outcomes = pd.DataFrame.from_dict({'event': events, 'duration': outcomes})
    outcomes['cluster'] = z

    return pd.DataFrame(x_signal), outcomes['duration'], outcomes['event'], parameters, outcomes

def compute_cif(x, betas, z, times):
    # causes and dim are read back out of betas, not re-specified here,
    # so this always matches whatever generate() was actually called with.
    causes = sorted(k for k in betas if k != -1)
    dim = betas[-1].shape[1]
    shapes = _build_shapes(dim)

    x_signal = x.values[:, :dim]
    shape_x = {event: shapes[event](betas[event][z], x_signal) for event in [-1] + causes}
    total_hazard = sum(shape_x[r] for r in causes)
    sc = shape_x[-1]

    # CIF for each cause = (relative hazard) x (CDF of the overall, all-cause event time).
    # Same c = total/w_s, scale = 1/w_s mapping as generate(), so the CDF here matches
    # the hazard lambda(t) = total_hazard * exp(w_s * t) exactly.
    cif = {}
    for event in causes:
        hazard_ratio = shape_x[event] / total_hazard
        cif[event] = pd.DataFrame(
            np.vstack([r * gompertz.cdf(times, c=tot / s, scale=1 / s)
                       for r, tot, s in zip(hazard_ratio, total_hazard, sc)]),
            columns=times,
        )
    return pd.concat(cif, axis=1)

def compute_cif_marginal_gap(x, betas, z, times, cause_compute = [1]):
    """
    For each cause r, compares:
      competing:     P(T' <= t, D' = r)   the actual CIF, accounting for competing risks
      non_competing: P(T_r <= t)          the probability if r were the only possible event
      diff:          competing - non_competing
 
    Both quantities use the same shared scale w_s; non_competing replaces the
    total hazard with the cause's own w_r alone (no other cause can remove
    someone from the risk set first).
 
    Returns a dict of three CIF-shaped DataFrames (MultiIndex columns: cause, time).
    """
    causes = sorted(k for k in betas if k != -1)
    signal_dim = betas[-1].shape[1]
    shapes = _build_shapes(signal_dim)
 
    x_signal = x.values[:, :signal_dim]
    shape_x = {event: shapes[event](betas[event][z], x_signal) for event in [-1] + causes}
    total_hazard = sum(shape_x[r] for r in causes)
    sc = shape_x[-1]
 
    competing, non_competing = {}, {}
    for event in cause_compute:
        hazard_ratio = shape_x[event] / total_hazard
        competing[event] = pd.DataFrame(
            np.vstack([hr * gompertz.cdf(times, c=tot / s, scale=1 / s)
                       for hr, tot, s in zip(hazard_ratio, total_hazard, sc)]),
            columns=times, index=x.index,
        )
        non_competing[event] = pd.DataFrame(
            np.vstack([gompertz.cdf(times, c=wr / s, scale=1 / s)
                       for wr, s in zip(shape_x[event], sc)]),
            columns=times, index=x.index,
        )
 
    competing = pd.concat(competing, axis=1)
    non_competing = pd.concat(non_competing, axis=1)
    return {'competing': competing, 'non_competing': non_competing, 'diff': non_competing - competing}
 