"""ESMFold2 group-COM angle + dihedral E2E test (qbp protein). SEPARATE from
restr_example.py so the canonical example is untouched. Run via run_grp_test.sh.

Verifies the NEW group-COM restraints on the real ESMFold2 diffusion head: angle of 3
region COMs -> 90 deg (vertex pinned), dihedral of 4 region COMs -> 180 deg (axis pinned).
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

RESTRAINTS_CONFIG = {
    "verbose": True,
    "gpu": True,
    "max_iter": 1000,
    "method": "CG",
    "angle_restraints_config": [
        {
            "atom_selection1": "chain A and (resid 5 to 84)",
            "atom_selection2": "chain A and (resid 90 to 180)",  # vertex (pinned)
            "atom_selection3": "chain A and (resid 186 to 224)",
            "harmonic": {"target_angle": 90.0},
            "start_sigma": 99999999,
        }
    ],
    "dihedral_restraints_config": [
        {
            "atom_selection1": "chain A and (resid 5 to 50)",
            "atom_selection2": "chain A and (resid 51 to 100)",  # axis (pinned)
            "atom_selection3": "chain A and (resid 101 to 150)",  # axis (pinned)
            "atom_selection4": "chain A and (resid 151 to 224)",
            "harmonic": {"target_dihedral": 180.0},
            "start_sigma": 99999999,
        }
    ],
}


def main() -> None:
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda()
    model.train(False)
    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=QBP)])
    result = ESMFold2InputBuilder().fold(
        model, spi, num_loops=3, num_sampling_steps=200, seed=0,
        restraints_config=RESTRAINTS_CONFIG,
    )
    with open("out_grp_test.cif", "w") as fh:
        fh.write(result.complex.to_mmcif())
    print("wrote out_grp_test.cif")


if __name__ == "__main__":
    main()
