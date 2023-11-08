import numpy as np
import pandas as pd
from scipy.stats import gompertz, bernoulli
from sklearn.datasets import make_blobs
def generate_old(random_seed = 42, size = 10000, competing_risks = 2):
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

def compute_cif_old(x, betas, z, times):
    cif = {}

    for event in range(len(betas)):
        # The cif in closed form is a compretz with c1+c2 parameter (c1 being the parameter for the current risk and c2 for the competing)
        # Futher normalised by c1/ (c1+c2)
        c1 = np.exp((betas[event][z] * x.drop(columns = 'z')).sum(1))
        c2 = np.exp((betas[1 - event][z] * x.drop(columns = 'z')).sum(1))
        cif[event + 1] = pd.DataFrame(np.vstack([c1_ / (c1_ + c2_) * gompertz.cdf(times, c1_ + c2_) for c1_, c2_ in zip(c1, c2)]), columns = times)

    return pd.concat(cif, axis = 1)


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
    parameters = {event: np.array([np.random.normal(size = 12) for _ in np.unique(z)]) / np.linspace(5, 10, 12) for event in range(3)}
    parameters[0] /= 5

    # Generate the data with the summed hazard
    shape = np.exp((parameters[1][z] * x).sum(1)) + np.exp((parameters[2][z] * x).sum(1)) 
    scale = np.exp((parameters[0][z] * x).sum(1))
    outcomes = gompertz.rvs(shape, scale = scale, random_state = random_seed)

    # Assign the outcomes following Bernoulli draw
    hazard_1 = np.exp((parameters[1][z] * x).sum(1))
    hazard_2 = np.exp((parameters[2][z] * x).sum(1))
    hazard_ratio = hazard_1 / (hazard_1 + hazard_2)
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
    for event in [1, 2]:
        shape = np.exp((betas[event][z] * x).sum(1))
        scale = np.exp((betas[0][z] * x).sum(1))

        hazard_1 = np.exp((betas[event][z] * x).sum(1))
        hazard_2 = np.exp((betas[3 - event][z] * x).sum(1))
        hazard_ratio = hazard_1 / (hazard_1 + hazard_2)

        cif[event] = pd.DataFrame(np.vstack([r * gompertz.cdf(times, sh, scale = sc) for r, sh, sc in zip(hazard_ratio, shape, scale)]), columns = times)

    return pd.concat(cif, axis = 1)
