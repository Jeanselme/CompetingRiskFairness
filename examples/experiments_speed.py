
# Comparsion models for competing risks
# In this script we train the different models for competing risks
import sys
from generate import *
from experiment import *

# Select random seed
random_seed = int(sys.argv[1])

print("Script running experiments on generated data with seed =", random_seed)
x, t, e, _, _ = generate(random_seed) 

# Normalise data
x, t, e = StandardScaler().fit_transform(x.values).astype(float),\
            t.values.astype(float),\
            e.values.astype(int)

# Hyperparameters
max_epochs = 1000
grid_search = 1
batch = [500]

for n in [2, 5]:
    param_grid = {
        'epochs': [max_epochs],
        'learning_rate' : [1e-3],
        'batch': batch,

        'k' : [n],
        'distribution' : ['Weibull'],
        'layers' : [[50] * 6],
    }
    DSMExperiment.create(param_grid, k = 1, n_iter = grid_search, path = 'Results_speed/generate={}_dsm{}'.format(random_seed, n), random_seed = random_seed).train(x, t, e)

# NFG Competing risk
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3],
    'batch': batch,
    'patience_max': [2],
    'multihead': [True],
    'layers_surv': [[50] * 3],
    'layers': [[50] * 3], 
    'act': ['Tanh']
}
NFGExperiment.create(param_grid, k = 1, n_iter = grid_search, path = 'Results_speed/generate={}_nfg'.format(random_seed), random_seed = random_seed).train(x, t, e)

# Desurv
for n in [1, 5, 15]:
    param_grid = {
        'epochs': [max_epochs],
        'learning_rate' : [1e-3],
        'batch': batch,
        'n': [n],
        'multihead': [True],
        'patience_max': [2],
        'layers_surv': [[50] * 3],
        'layers': [[50] * 3], 
        'act': ['Tanh']
    }
    DeSurvExperiment.create(param_grid, k = 1, n_iter = grid_search, path = 'Results_speed/generate={}_ds{}'.format(random_seed, n), random_seed = random_seed).train(x, t, e)

# DeepHit Competing risk
for n in [15, 100]:
    param_grid = {
        'epochs': [max_epochs],
        'learning_rate' : [1e-3],
        'batch': batch,
        'n': [n],
        'nodes' : [[50] * 3],
        'shared' : [[50] * 3]
    }
    DeepHitExperiment.create(param_grid, k = 1, n_iter = grid_search, path = 'Results_speed/generate={}_dh{}'.format(random_seed, n), random_seed = random_seed).train(x, t, e)