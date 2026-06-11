"""ESMFold2 RGI (restraint-guided inference) example via rgi_utils.

ESMFold2 has a diffusion structure head, so RGI plugs in exactly like the other
*_restr tools: a ``restraints_config`` dict is passed to ``fold()``, and the model
minimizes the restraint energy on the x0 prediction after each diffusion step.

Run (GPU; install the local model fork + rgi_utils first, e.g.):
    pixi run python -m pip install -e ../transformers_restr -e "../rgi_utils[torch]"
    pixi run python restr_example.py

This folds the QBP (glutamine-binding protein) with its natural ligand GLN and the
shared RGI restraint set: a harmonic distance restraint (target 25 A) between two
protein residue groups + a conformer restraint holding GLN at its ideal geometry.
This is the same example as the other *_restr tools, in ESMFold2's Python-API form.
"""

from __future__ import annotations

from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    LigandInput,
    ProteinInput,
    StructurePredictionInput,
)

QBP = (
    "ADKKLVVATDTAFVPFEFKQGDKYVGFDVDLWAAIAKELKLDYELKPMDFSGIIPALQTKNVDLALAGITITDERKK"
    "AIDFSDGYYKSGLLVMVKANNNDVKSVKDLDGKVVAVKSGTGSVDYAKANIKTKDLRQFPNIDNAYMELGTNRADAV"
    "LHDTPNILYFIKTAGNGQFKAVGDSLEAQQYGIAFPKGSDELRDKVNGALKTLRENGTYNEIYKKWFGTEPK"
)

# rgi_utils restraints_config — same schema as the other *_restr tools.
# ``chain`` / ``resid`` use the per-chain 1-based residue ordinal (resets per chain;
# a ligand atom gets its own ordinal). start_sigma is OPTIONAL per distance entry and
# once in conformer_restraints_config (omit -> active every step, i.e. when
# sigma <= start_sigma); there is no global/top-level start_sigma.
# stop_sigma is the LOWER-bound counterpart (release once sigma < stop_sigma; window
# stop_sigma <= sigma <= start_sigma) on EVERY restraint type, default -1 = off; set it
# (e.g. 1.0) on the RMSD restraint to fix a peptide bond broken at a restrained /
# unmodeled-tail boundary.
RESTRAINTS_CONFIG = {
    "verbose": True,
    "gpu": True,  # torch device: True=GPU, False=CPU
    "max_iter": 1000,
    "method": "CG",
    "distance_restraints_config": [
        {
            # COM distance between two protein residue groups, pulled to 25 A. Qualify
            # with `chain A` so a bare resid range cannot also sweep in the ligand
            # (chain B). type: harmonic{target_distance} |
            # flat-bottomed{target_distance1,target_distance2} | flat-bottomed1 | flat-bottomed2.
            "atom_selection1": "chain A and ((resid 5 to 84) or (resid 186 to 224))",
            "atom_selection2": "chain A and (resid 90 to 180)",
            "start_sigma": 99999999,  # optional; omit -> every step (this value == always on)
            # "stop_sigma": -1,  # optional lower bound (default -1 = off)
            # "move": "both",  # which group moves: both (default) / 1 (only
            #   atom_selection1) / 2 (only atom_selection2); 1/2 pin the other group.
            "harmonic": {"target_distance": 25.0},
        }
    ],
    "conformer_restraints_config": {
        # Hold the GLN ligand at its ideal RDKit geometry (bond/angle/chiral + intramol
        # VdW; GLN has one stereocentre at CA). ESMFold2 carries ligand bond ORDERS on
        # ChainInfo.ligand_bond_orders (CCD via get_ligand_ccd_bonds, SMILES via Kekulized
        # 3-tuples), so dihedral (cis/trans) works for BOTH — GLN just has none (terminal
        # C=O) so dihedrals=0 here; use a fumarate/maleate ligand to exercise it.
        "start_sigma": 99999999,  # optional; ONE value for ALL conformer terms (omit -> every step)
        # "stop_sigma": -1,  # lower bound for ALL conformer terms (default -1 = off)
        "bond": {"weight": 1.0},
        "angle": {"weight": 1.0},
        "chiral": {"weight": 1.0},
        "dihedral": {"weight": 1.0},
        "vdw": {"weight": 1.0, "mode": "intramolecular"},
    },
    # --- RMSD restraint ----------------------------------------------------------
    # Kabsch-superposed RMSD of a moving group toward target_rmsd vs a reference PDB.
    # REQUIRES a reference: fold this sequence once WITHOUT restraints, write
    # result.complex.to_mmcif(), convert to rmsd_ref.pdb (gemmi make_pdb_string()) first.
    # Atoms pair by IDENTITY (chain, resid, name), so the ref atom order need not match.
    # Every key is shown; uncomment the optional (commented) ones as needed.
    "rmsd_restraints_config": [
        {
            "ref_pdb": "rmsd_ref.pdb",   # REQUIRED: path to the reference PDB
            "target_rmsd": 0.0,          # REQUIRED: target Kabsch-RMSD in A (0 = match ref)
            # "weight": 1.0,             # optional (default 1.0)
            # "start_sigma": 99999999,   # optional (omit -> every step; 1.0 = late only)
            # "stop_sigma": 1.0,         # optional LOWER bound (default -1 = off):
            #   RELEASE below this sigma so the model re-idealises geometry it held
            #   distorted — fixes a broken peptide bond at a restrained-residue /
            #   unmodeled-tail boundary; 1.0 validated for this whole-structure case.
            # --- residue correspondence (optional) ---
            # "pairing": "identity",     # "identity" (default) | "align" = sequence-align
            #                            #   polymer chains (BLOSUM62) for a HOMOLOG ref
            #                            #   (tolerates substitutions/indels/renumbering).
            #                            #   NOTE align pairs backbone + ANY shared side-chain
            #                            #   atom names by default, pinning a substituted side
            #                            #   chain; add a `name CA` atom_selection for backbone.
            # "best_effort": True,       # default True (skip atoms unmatched in the ref);
            #                            #   False = strict (raise on any unmatched atom)
            # --- atom selection (optional; omit -> WHOLE structure, best-effort) ---
            # "atom_selection_ref": "chain A and (resid 90 to 180)",     # shorthand: sets
            # "atom_selection_target": "chain A and (resid 90 to 180)",  #   BOTH fit & calc
            # --- OR split superposition (fit) vs measured (calc) atoms ---
            # "atom_selection_ref_fit": "chain A and (resid 5 to 220)",
            # "atom_selection_target_fit": "chain A and (resid 5 to 220)",
            # "atom_selection_ref_calc": "chain A and (resid 90 to 180)",
            # "atom_selection_target_calc": "chain A and (resid 90 to 180)",
        },
    ],
}


def main() -> None:
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda()
    model.train(False)  # inference / eval mode

    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id="A", sequence=QBP),
            LigandInput(id="B", ccd=["GLN"]),  # glutamine — QBP's natural ligand
        ]
    )

    result = ESMFold2InputBuilder().fold(
        model,
        spi,
        num_loops=3,
        num_sampling_steps=200,
        seed=0,
        restraints_config=RESTRAINTS_CONFIG,
    )
    with open("out_esm.cif", "w") as fh:
        fh.write(result.complex.to_mmcif())
    print("wrote out_esm.cif")


if __name__ == "__main__":
    main()
