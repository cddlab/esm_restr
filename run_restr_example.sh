#!/bin/bash
#SBATCH -J esm_ex
#SBATCH -o run_restr_example.out
#SBATCH -e run_restr_example.err
#SBATCH -p q1
#SBATCH --gres=gpu:1
# ESMFold2 RGI example runner (pixi env). GPU work must go through sbatch.
# Submit from THIS repo directory:  cd esm_restr && sbatch run_restr_example.sh
# Run on q1/q3 (RTX 4090, sm_89) — this pixi env's torch has no Blackwell (sm_120)
# kernels, so maxq/q4 fail at model init.
set -e
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Prefer a fresh pixi binary if the workspace ships one, else fall back to PATH.
PIXI="${PIXI:-$([ -x ../.pixi-bin/pixi ] && echo ../.pixi-bin/pixi || echo pixi)}"

# Build the env, then install the LOCAL transformers fork (the esmfold2 model with the RGI
# hook) and rgi_utils editable so both override the pyproject git deps.
"$PIXI" install
"$PIXI" run python -m pip install -e ../transformers_restr -e "../rgi_utils[torch]"

# restr_example.py folds qbp with a single harmonic distance restraint (target 25 A) and a
# commented-out ligand-conformer variant. RGI: restraints_config is a Python dict passed to
# ESMFold2InputBuilder().fold(); the model minimizes it on the x0 prediction each step.
"$PIXI" run python restr_example.py

# Verify the two restraint groups' COM distance reaches ~25 A (root check script).
"$PIXI" run python ../check_dist.py out_esm_dist.cif || true
echo done
