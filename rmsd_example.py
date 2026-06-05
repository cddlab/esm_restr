"""ESMFold2 RGI RMSD-restraint example via rgi_utils.

An RMSD restraint drives the Kabsch-superposed RMSD of a moving atom group
(atom_selection_target) toward ``target_rmsd``, measured against a reference PDB
(ref_pdb + atom_selection_ref); it is minimized by the CG solver after each
diffusion step, like the other restraint types.

Run (GPU; install the local model fork + rgi_utils first):
    pixi run python -m pip install -e ../transformers_restr -e "../rgi_utils[torch]"
    # 1) Generate a reference PDB (e.g. fold this sequence once without restraints,
    #    write result.complex.to_mmcif(), then gemmi make_pdb_string()) -> rmsd_ref.pdb
    # 2) pixi run python rmsd_example.py

Atoms pair by selection ORDER and the two selections MUST pick the same number of
atoms (a mismatch is a hard error). The selection DSL is chain/resid/index only.
"""

from __future__ import annotations

from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    ProteinInput,
    StructurePredictionInput,
)

QBP = (
    "ADKKLVVATDTAFVPFEFKQGDKYVGFDVDLWAAIAKELKLDYELKPMDFSGIIPALQTKNVDLALAGITITDERKK"
    "AIDFSDGYYKSGLLVMVKANNNDVKSVKDLDGKVVAVKSGTGSVDYAKANIKTKDLRQFPNIDNAYMELGTNRADAV"
    "LHDTPNILYFIKTAGNGQFKAVGDSLEAQQYGIAFPKGSDELRDKVNGALKTLRENGTYNEIYKKWFGTEPK"
)

# rgi_utils restraints_config — same schema as the other *_restr tools.
RESTRAINTS_CONFIG = {
    "verbose": True,
    "gpu": True,  # -> torch backend (ESMFold2 runs on GPU)
    "max_iter": 200,
    "method": "CG",
    "rmsd_restraints_config": [
        {
            # Drive residues 90-180 to match the reference (RMSD -> 0).
            "ref_pdb": "rmsd_ref.pdb",
            "target_rmsd": 0.0,
            "atom_selection_ref": "chain A and (resid 90 to 180)",
            "atom_selection_target": "chain A and (resid 90 to 180)",
            "start_sigma": 99999999,  # optional; omit -> active every step
        }
    ],
}


def main() -> None:
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda()
    model.train(False)

    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=QBP)])
    result = ESMFold2InputBuilder().fold(
        model,
        spi,
        num_loops=3,
        num_sampling_steps=200,
        seed=1,
        restraints_config=RESTRAINTS_CONFIG,
    )
    with open("out_esm_rmsd.cif", "w") as fh:
        fh.write(result.complex.to_mmcif())
    print("wrote out_esm_rmsd.cif")


if __name__ == "__main__":
    main()
