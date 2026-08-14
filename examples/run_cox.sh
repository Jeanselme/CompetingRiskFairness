#!/bin/bash
set -euo pipefail

# Usage: ./run_cox.sh <causes> <dim>
#   causes : number of competing risks (2 or 3)
#   dim    : number of covariate dimensions (20 or 30)

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <causes> <dim>"
    echo "  causes: 2 or 3"
    echo "  dim:    20 or 30"
    exit 1
fi

CAUSES=$1
DIM=$2

# Validate arguments
case "$CAUSES" in 2|3)   ;; *) echo "Error: causes must be 2 or 3 (got '$CAUSES')"; exit 1 ;; esac
case "$DIM"    in 20|30) ;; *) echo "Error: dim must be 20 or 30 (got '$DIM')";    exit 1 ;; esac

source /opt/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PWD:$PWD/NeuralFineGray:$PWD/NeuralFineGray/DeepSurvivalMachines:${PYTHONPATH:-}"
conda activate survival

for seed in {0..24}; do
    echo "=== Cox/Fine-Gray: causes=$CAUSES, dim=$DIM, seed=$seed ==="

    # Extract data for this seed, passing causes and dim so the R model
    # sees the same generated dataset as the Python models.
    python examples/process_data.py -seed "$seed" --cause "$CAUSES" --dim "$DIM"

    # Run all R models (Cox / Fine-Gray)
    Rscript examples/FineGray.R
    rm -f tmp.csv

    # Move each model output to Results using the Python naming convention:
    #   generate_causes=<C>_seed=<S>_dim=<D>_<suffix>
    find . -maxdepth 1 -type f -name 'tmp*' | while read -r file; do
        suffix="${file##*_}"   # original suffix of the tmp file
        mv "$file" "Results/generate_causes=${CAUSES}_seed=${seed}_dim=${DIM}_${suffix}"
        echo "  seed $seed completed - $suffix"
    done
done