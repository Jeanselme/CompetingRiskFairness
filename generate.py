import numpy as np
import pandas as pd
from scipy.stats import gompertz
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def generate(random_seed = 42, size = 10000, competing_risks = 2):
    """
        Generate two clusters with different survival profiles
        Concatenate clusters and covariates for X
        Model two different outcomes

        return x, t, e, betas, z
    """
    np.random.seed(random_seed)
    # Data - Generate two blobs
    x, z = make_blobs(n_samples = size, n_features = 2, centers = ([-1.5, -1.5], [1.5, 1.5]), random_state = random_seed)
    x = np.column_stack([x] + [np.random.normal(size = size) for _ in range(10)]) 

    betas = {event: np.array([np.random.normal(size = 12) for _ in np.unique(z)]) / np.linspace(5, 10, 12) for event in range(competing_risks)}

    # Outcome
    outcomes = {
        event: np.zeros(size) for event in range(competing_risks + 1)   
    }

    # Create the different event times
    for event in range(competing_risks):
        params = np.exp((betas[event][z] * x).sum(1))
        # Ensure positivity parameters
        outcomes[event + 1] = gompertz.rvs(params, random_state = random_seed)

    # Create censoring
    outcomes[0] = gompertz.rvs(np.exp(np.random.random(len(x))) / 2, random_state = random_seed)

    outcomes = pd.DataFrame.from_dict(outcomes)
    outcomes['duration'], outcomes['event'] = outcomes.min(axis = 1), outcomes.idxmin(axis = 1)
    outcomes['cluster'] = z

    return pd.concat([pd.DataFrame(x), pd.Series(z, name = 'z')], axis = 1), outcomes['duration'], outcomes['event'], betas, outcomes

def compute_cif(x, betas, z, times):
    cif = {}

    for event in range(len(betas)):
        # The cif in closed form is a compretz with c1+c2 parameter (c1 being the parameter for the current risk and c2 for the competing)
        # Futher normalised by c1/ (c1+c2)
        c1 = np.exp((betas[event][z] * x.drop(columns = 'z')).sum(1))
        c2 = np.exp((betas[1 - event][z] * x.drop(columns = 'z')).sum(1))
        cif[event + 1] = pd.DataFrame(np.vstack([c1_ / (c1_ + c2_) * gompertz.cdf(times, c1_ + c2_) for c1_, c2_ in zip(c1, c2)]), columns = times)

    return pd.concat(cif, axis = 1)