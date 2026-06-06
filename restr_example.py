"""ESMFold2 RGI (restraint-guided inference) example via rgi_utils.

ESMFold2 has a diffusion structure head, so RGI plugs in exactly like the other
*_restr tools: a ``restraints_config`` dict is passed to ``fold()``, and the model
minimizes the restraint energy on the x0 prediction after each diffusion step.

Run (GPU; install the local model fork + rgi_utils first, e.g.):
    pixi run python -m pip install -e ../transformers_restr -e "../rgi_utils[torch]"
    pixi run python restr_example.py

This folds the qbp protein with a single harmonic distance restraint (target 25 A)
between two residue groups — the shared RGI smoke fixture. A ligand-conformer variant
is shown commented out below (add a LigandInput and the conformer block).
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
# ``chain`` / ``resid`` use the per-chain 1-based residue ordinal (resets per chain).
RESTRAINTS_CONFIG = {
    "verbose": True,
    "gpu": True,  # torch device: True=GPU, False=CPU
    "max_iter": 200,
    "method": "CG",
    # start_sigma is OPTIONAL per distance entry and in conformer_restraints_config
    # (omit -> active every step; active when sigma <= start_sigma); no global/top-level.
    "distance_restraints_config": [
        {
            "atom_selection1": "chain A and ((resid 5 to 84) or (resid 186 to 224))",
            "atom_selection2": "chain A and (resid 90 to 180)",
            "start_sigma": 99999999,
            "harmonic": {"target_distance": 25.0},
        }
    ],
    # --- Ligand conformer variant ------------------------------------------------
    # Add a ligand chain (e.g. LigandInput(id="B", ccd=["ATP"])) to the input below
    # and uncomment to hold the ligand at its CCD reference geometry. NOTE: ESMFold2
    # ligand bonds come from token_bonds (connectivity only, no bond order), so the
    # dihedral (cis/trans) term finds nothing here (dihedrals=0), like chai.
    # "conformer_restraints_config": {
    #     "start_sigma": 1.0,   # optional (omit -> every step)
    #     "bond": {"weight": 1.0},
    #     "angle": {"weight": 1.0},
    #     "chiral": {"weight": 1.0},
    #     "dihedral": {"weight": 1.0},
    #     "vdw": {"weight": 1.0, "mode": "intramolecular"},
    # },
    # --- RMSD restraint ----------------------------------------------------------
    # Kabsch-superposed RMSD of a moving group toward target_rmsd vs a reference PDB.
    # REQUIRES a reference: fold this sequence once WITHOUT restraints, write
    # result.complex.to_mmcif(), convert to rmsd_ref.pdb (gemmi make_pdb_string()) first.
    # Atoms pair by IDENTITY (chain, resid, name), so the ref atom order need not match.
    # atom_selection is OPTIONAL: omitted here -> fit + measure RMSD over the WHOLE structure,
    # best-effort (atoms missing from the ref are skipped); give atom_selection_ref/target
    # (or _fit/_calc) to restrain a region.
    "rmsd_restraints_config": [
        {"ref_pdb": "rmsd_ref.pdb", "target_rmsd": 0.0},  # whole structure; for a region
        # add "atom_selection_ref"/"atom_selection_target": "chain A and (resid 90 to 180)"
    ],
}


def main() -> None:
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda()
    model.train(False)  # inference / eval mode

    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=QBP)])
    # Conformer example: append LigandInput(id="B", ccd=["ATP"]) to the sequences.

    result = ESMFold2InputBuilder().fold(
        model,
        spi,
        num_loops=3,
        num_sampling_steps=200,
        seed=0,
        restraints_config=RESTRAINTS_CONFIG,
    )
    with open("out_esm_dist.cif", "w") as fh:
        fh.write(result.complex.to_mmcif())
    print("wrote out_esm_dist.cif")


if __name__ == "__main__":
    main()
