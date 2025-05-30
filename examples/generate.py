import numpy as np
import pandas as pd
from scipy.stats import gompertz, bernoulli
from sklearn.datasets import make_blobs

shape = {
        -1: lambda p, x: np.abs((p * x)[:, 5:]).sum(1) / 10,
        0 : lambda p, x: ((p * x)[:, 5:] ** 2).sum(1) / 10,
        1 : lambda p, x: np.abs(((p * x)[:, 5:] ** 2) + ((p * x)[:, :5])).sum(1)  / 10,
        2 : lambda p, x: np.abs(((p * x)[:, :5] ** 2) + ((p * x)[:, 5:])).sum(1)  / 10
    }

def generate(random_seed = 42, size = 30000):
    """
        Generate two clusters with different survival profiles
        Concatenate clusters and covariates for X
        Model two different outcomes

        return x, t, e, betas, z
    """
    # Set seed for the experiment
    np.random.seed(random_seed)

    # Data - Generate two blobs
    x, z = make_blobs(n_samples = size, n_features = 2, centers = ([-1.5, -1.5], [1.5, 1.5]))
    x = np.column_stack([x] + [np.random.normal(size = size) for _ in range(8)]) 

    # Generate parameters for each gompretz cause specific hazards
    parameters = {event: np.array([np.random.normal(size = 10) for _ in np.unique(z)]) for event in [-1, 1, 2]}

    # Generate the data with the summed hazard
    s1 = shape[1](parameters[1][z], x)
    s2 = shape[2](parameters[2][z], x)
    sc = shape[-1](parameters[-1][z], x)
    outcomes = gompertz.rvs(s1 + s2, sc)

    # Assign the outcomes following Bernoulli draw
    hazard_ratio = s1 / (s1 + s2)
    events = 2 - bernoulli.rvs(hazard_ratio)

    # Create censoring NON RANDOM
    censoring_beta = np.random.normal(size = 10)
    censoring = gompertz.rvs(shape[0](censoring_beta, x))
    events = (censoring > outcomes) * events
    outcomes = (censoring > outcomes) * outcomes + (censoring <= outcomes) * censoring

    outcomes = pd.DataFrame.from_dict({'event': events, 'duration': outcomes})
    outcomes['cluster'] = z

    return pd.DataFrame(x), outcomes['duration'], outcomes['event'], parameters, outcomes

def compute_cif(x, betas, z, times):
    # As we know each cause specific hazard, we can model the associated gompretz
    cif = {}
    shape_x = {event: shape[event](betas[event][z], x.values) for event in [-1, 1, 2]}
    for event in [1, 2]:
        hazard_ratio = shape_x[event] / (shape_x[1] + shape_x[2])
        cif[event] = pd.DataFrame(np.vstack([r * gompertz.cdf(times, sh, sc) for r, sh, sc in zip(hazard_ratio, shape_x[event], shape_x[-1])]), columns = times)

    return pd.concat(cif, axis = 1)

def conditional_survival(w1, w2, ws, times, t_eval):
    max_time = np.searchsorted(times, t_eval)

    F2 = gompertz.cdf(times[:max_time], w2, ws)
    f1 = gompertz.pdf(times[:max_time], w1, ws)

    numerator = np.trapz(F2 * f1, times[:max_time])
    denominator = gompertz.cdf(t_eval, w1, ws)
    return numerator / denominator if denominator != 0 else 0

def conditional_probability(x, betas, z, times, t_eval):
    shape_x = {event: shape[event](betas[event][z], x.values) for event in [-1, 1, 2]}
    return pd.DataFrame({t: np.array([
        conditional_survival(w1, w2, ws, times, t) for w1, w2, ws in zip(shape_x[1], shape_x[2], shape_x[-1])]) 
        for t in t_eval}, index = x.index)


