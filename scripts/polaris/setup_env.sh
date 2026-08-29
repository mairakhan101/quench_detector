#!/bin/bash
# Setup conda environment for quench_detector on Polaris (ALCF)
#
# Run once:  bash scripts/polaris/setup_env.sh
# Activate:  export PATH=/soft/applications/conda/2024-04-29/mconda3/bin:$PATH && \
#            eval "$(conda shell.bash hook)" && conda activate quench_surr

set -euo pipefail

export PATH=/soft/applications/conda/2024-04-29/mconda3/bin:$PATH
eval "$(conda shell.bash hook)"

ENV_NAME="quench_surr"

if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists — updating"
    conda activate "$ENV_NAME"
    pip install -r requirements.txt
else
    echo "Creating environment '$ENV_NAME'"
    conda create -n "$ENV_NAME" python=3.11 -y
    conda activate "$ENV_NAME"
    pip install -r requirements.txt
    pip install "jax[cpu]==0.4.29"
fi

echo "Done. Activate with:  conda activate $ENV_NAME"
