# Comparison models for competing risks
# In this script we train the different models for competing risks
import sys
sys.path.append('NeuralFineGray/')
sys.path.append('NeuralFineGray/DeepSurvivalMachines/')

import argparse
from generate import *
from experiment import *

parser = argparse.ArgumentParser(description='Train competing risk models on generated data.')
parser.add_argument('--causes', type=int, default=2,        help='Number of competing risks (default: 2)')
parser.add_argument('--seed',   type=int, default=None,     help='Single random seed to run (default: run loop 0-24)')
parser.add_argument('--dim',    type=int, default=10,       help='Number dimensions (default: 10)')
args = parser.parse_args()

seeds = [args.seed] if args.seed is not None else range(25)

# Hyperparameters
max_epochs  = 1000
grid_search = 100
layers      = [[i] * (j + 1) for i in [25, 50] for j in range(4)]
batch       = [1000]

for random_seed in seeds:
    print(f"Running experiments with n_causes={args.causes}, seed={random_seed}")
    x, t, e, _, _ = generate(random_seed, causes=list(range(1, args.causes + 1)), dim=args.dim)

    # Normalise data
    x, t, e = (StandardScaler().fit_transform(x.values).astype(float),
                t.values.astype(float),
                e.values.astype(int))

    def run(Experiment, suffix, events, param_grid):
        path = f'Results/generate_causes={args.causes}_seed={random_seed}_dim={args.dim}_{suffix}'
        Experiment.create(param_grid, k=1, n_iter=grid_search, path=path,
                          delete_log=True, random_seed=random_seed).train(x, t, events)

    # NFG
    param_grid = {
        'epochs':        [max_epochs],
        'learning_rate': [1e-3, 1e-4],
        'batch':         batch,
        'multihead':     [True],
        'layers_surv':   layers,
        'layers':        layers,
        'act':           ['Tanh'],
        'normalise':     ['minmax']
    }
    run(NFGExperiment,     'nfg',   e, param_grid)
    run(NFGExperiment,     'nfgnc', (e == 1).astype(int), param_grid)

    # DeSurv
    param_grid = {
        'epochs':        [max_epochs],
        'learning_rate': [1e-3, 1e-4],
        'batch':         batch,
        'embedding':     [True],
        'multihead':     [True],
        'layers_surv':   layers,
        'layers':        layers,
        'act':           ['Tanh'],
        'normalise':     ['minmax']
    }
    run(DeSurvExperiment,  'ds',   e, param_grid)
    run(DeSurvExperiment,  'dsnc', (e == 1).astype(int), param_grid)

    # DeepHit
    param_grid = {
        'epochs':        [max_epochs],
        'learning_rate': [1e-3, 1e-4],
        'batch':         batch,
        'nodes':         layers,
        'shared':        layers,
    }
    run(DeepHitExperiment, 'dh',   e, param_grid)
    run(DeepHitExperiment, 'dhnc', (e == 1).astype(int), param_grid)