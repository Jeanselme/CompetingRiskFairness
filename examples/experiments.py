
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
grid_search = 100
layers = [[i] * (j + 1) for i in [25, 50] for j in range(4)]
layers_large = [[i] * (j + 1) for i in [25, 50] for j in range(8)]
batch = [1000]

# DSM
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,

    'k' : [2, 3, 4, 5],
    'distribution' : ['LogNormal', 'Weibull'],
    'layers' : layers_large,
}
DSMExperiment.create(param_grid, k = 1, n_iter = grid_search, path = 'Results/generate={}_dsm'.format(random_seed), delete_log =True, random_seed = random_seed).train(x, t, e)
DSMExperiment.create(param_grid, k = 1, n_iter = grid_search, path = 'Results/generate={}_dsmnc'.format(random_seed), delete_log =True, random_seed = random_seed).train(x, t, (e == 1).astype(int))

# NFG Competing risk
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,
    'dropout': [0, 0.25, 0.5, 0.75],

    'multihead': [True],
    'layers_surv': layers,
    'layers' : layers,
    'act': ['Tanh']
}
NFGExperiment.create(param_grid, k = 1, n_iter = grid_search, path = 'Results/generate={}_nfg'.format(random_seed), delete_log =True, random_seed = random_seed).train(x, t, e)
NFGExperiment.create(param_grid, k = 1, n_iter = grid_search, path = 'Results/generate={}_nfgcs'.format(random_seed), delete_log =True, random_seed = random_seed).train(x, t, e, cause_specific = True)
NFGExperiment.create(param_grid, k = 1, n_iter = grid_search, path = 'Results/generate={}_nfgnc'.format(random_seed), delete_log =True, random_seed = random_seed).train(x, t, (e == 1).astype(int))

# Desurv
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,
    'embedding': [True], # To ensure same architecture

    'layers_surv': layers,
    'layers': layers,
    'act': ['Tanh'],
}
DeSurvExperiment.create(param_grid, k = 1, n_iter = grid_search, path = 'Results/generate={}_ds'.format(random_seed), delete_log =True, random_seed = random_seed).train(x, t, e)
DeSurvExperiment.create(param_grid, k = 1, n_iter = grid_search, path = 'Results/generate={}_dsnc'.format(random_seed), delete_log =True, random_seed = random_seed).train(x, t, (e == 1).astype(int))

# DeepHit Competing risk
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,

    'nodes' : layers,
    'shared' : layers
}
DeepHitExperiment.create(param_grid, k = 1, n_iter = grid_search, path = 'Results/generate={}_dh'.format(random_seed), delete_log =True, random_seed = random_seed).train(x, t, e)
DeepHitExperiment.create(param_grid, k = 1, n_iter = grid_search, path = 'Results/generate={}_dhnc'.format(random_seed), delete_log =True, random_seed = random_seed).train(x, t, (e == 1).astype(int))