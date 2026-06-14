#!/bin/bash
#SBATCH -J esm_grp
#SBATCH -o run_grp_test.out
#SBATCH -e run_grp_test.err
#SBATCH -p q3
#SBATCH --gres=gpu:1
# ESMFold2 group-COM angle/dihedral E2E. q1/q3 (RTX 4090) only — this pixi torch has no
# Blackwell (sm_120) kernels.  cd esm_restr && sbatch run_grp_test.sh
set -e
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PIXI="${PIXI:-$([ -x ../.pixi-bin/pixi ] && echo ../.pixi-bin/pixi || echo pixi)}"
"$PIXI" install
"$PIXI" run python -m pip install -e "../rgi_utils[torch]"
# pixi keeps a real (non-editable) transformers dir in site-packages; `pip install -e
# transformers_restr` creates no finder that can win over it (same package name), so the
# stale esmfold2 (without the per-step restraints.minimize hook) is what imports. Overwrite
# the esmfold2 subpackage in-place with the source -- the RGI changes are confined to it.
SP=$("$PIXI" run python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
cp -rf ../transformers_restr/src/transformers/models/esmfold2/. "$SP/models/esmfold2/"
find "$SP/models/esmfold2" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
# Fail fast if the hook still isn't the one being imported.
"$PIXI" run python -c "import transformers.models.esmfold2.modeling_esmfold2_common as m, inspect; assert 'restraints.minimize' in inspect.getsource(m), 'RGI hook missing in imported transformers: '+m.__file__; print('hook OK:', m.__file__)"

"$PIXI" run python grp_test.py > run_grp_test.log 2>&1 || { echo "esm FAILED:"; tail -n 40 run_grp_test.log; exit 1; }
grep -iE "built spec|setup:|finalize" run_grp_test.log || true
GP=../chai-lab_restr/.venv/bin/python
"$GP" ../check_angle.py out_grp_test.cif 5-84 90-180 186-224 || true
"$GP" ../check_dihedral.py out_grp_test.cif 5-50 51-100 101-150 151-224 || true
echo done
