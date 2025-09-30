# CompetingRiskFairness
This repository allows reproducing the results presented in [Competing Risks:
Impact on Risk Estimation and Algorithmic Fairness](https://arxiv.org/abs/2508.05435). This work demonstrates that the improper handling of competing risks, a common practice in the literature, leads to systematic bias in risk estimates with group-specific patterns.

## Reproduce paper's results
To set up the environment:  

0. Clone the repository with dependencies: `git clone git@github.com:Jeanselme/CompetingRiskFairness.git --recursive`
1. Create a conda environment with all necessary libraries: `pycox`, `lifelines`, `pysurvival`
2. Add path `export PYTHONPATH="$PWD:$PWD/DeepSurvivalMachines:$PYTHONPATH"`

To reproduce the paper's results on cardiovascular management:  

3. Run `NeuralFineGray/examples/experiment_competing_risk.py FRAMINGHAM` to run all models on the `FRAMINGHAM` dataset.
4. Analysis using `examples/Analysis FRAMINGHAM.ipynb` to measure performance and bias resulting from ignoring competing risks.

To reproduce the synthetic results:  

3. Run `examples/experiments.py` to run all models.
4. Analysis using `examples/Analysis Synthetic.ipynb` to measure performance and bias resulting from ignoring competing risks.


## Requirements
The model relies on `DeepSurvivalMachines`, `pytorch`, `numpy`, and `tqdm`.  
To run the set of experiments, `pycox`, `lifelines`, and `pysurvival` are necessary.
