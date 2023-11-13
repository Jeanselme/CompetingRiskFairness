import numpy as np
import pandas as pd
from scipy.stats import gompertz, bernoulli
from sklearn.datasets import make_blobs

shape1 = lambda p, x: ((p * x) ** 2).sum(1)
shape2 = lambda p, x: np.exp((p * x).sum(1))
scale =  lambda p, x: np.exp((p * x).sum(1))

def generate(random_seed = 42, size = 10000):
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

    # Generate parameters for each gompretz cause specific hazards
    # parameters[0] is used for scale of the gompretz shared across risk
    parameters = {event: np.array([np.random.normal(size = 12) for _ in np.unique(z)]) / 3 for event in range(3)}
    parameters[0] /= 5

    # Generate the data with the summed hazard
    s1 = shape1(parameters[1][z], x)
    s2 = shape2(parameters[2][z], x)
    sc = scale(parameters[0][z], x)
    outcomes = gompertz.rvs(s1 + s2, scale = sc, random_state = random_seed)

    # Assign the outcomes following Bernoulli draw
    hazard_ratio = s1 / (s1 + s2)
    events = 2 - bernoulli.rvs(hazard_ratio)

    # Create censoring
    censoring = gompertz.rvs(np.exp(np.random.random(len(x))), random_state = random_seed)
    events = (censoring > outcomes) * events
    outcomes = (censoring > outcomes) * outcomes + (censoring <= outcomes) * censoring

    outcomes = pd.DataFrame.from_dict({'event': events, 'duration': outcomes})
    outcomes['cluster'] = z

    return pd.concat([pd.DataFrame(x), pd.Series(z, name = 'z')], axis = 1), outcomes['duration'], outcomes['event'], parameters, outcomes

def compute_cif(x, betas, z, times):
    # As we know each cause specific hazard, we can model the associated gompretz
    cif = {}
    x = x.drop(columns = 'z')
    shape = {1: shape1(betas[1][z], x),
             2: shape2(betas[2][z], x)}
    sc = scale(betas[0][z], x)
    for event in [1, 2]:
        hazard_ratio = shape[event] / (shape[1] + shape[2])
        cif[event] = pd.DataFrame(np.vstack([r * gompertz.cdf(times, sh, scale = s) for r, sh, s in zip(hazard_ratio, shape[event], sc)]), columns = times)

    return pd.concat(cif, axis = 1)
