"""Tests for ESMFold2 input preparation (prepare_input)."""

import pytest
from rdkit import Chem

from esm.models.esmfold2.prepare_input import (
    build_chains_from_input,
    compute_token_bonds,
)
from esm.models.esmfold2.processor import ESMFold2InputBuilder
from esm.models.esmfold2.types import LigandInput, StructurePredictionInput


MAU_SMILES = (
    "C/C=C\\C=C\\[C@@H]1O[C@](O)([C@H](CC)C(=O)NC/C=C/C=C(\\C)"
    "[C@@H](OC)[C@@H](C)[C@@H]2O[C@H](/C=C/C=C/C=C(\\C)C(=O)"
    "c3c(O)ccn(C)c3=O)[C@H](O)[C@@H]2O)[C@H](O)[C@H](O)C1(C)C"
)


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1",  # benzene: 6 atoms, 6 bonds
        # The drug-like ligand from the SMILES-vs-CCD issue.
        "COC1=CC=C(N2C3=C(C(C(N)=O)=N2)CCN(C4=CC=C(N5CCCCC5=O)C=C4)C3=O)C=C1",
    ],
)
def test_smiles_ligand_bonds_match_molecular_graph(smiles: str):
    """SMILES ligand bonds must match the molecular graph, not a clique (#313)."""
    spi = StructurePredictionInput(sequences=[LigandInput(id="B", smiles=smiles)])
    chains, tokens, atoms = build_chains_from_input(spi, seed=0)
    token_bonds = compute_token_bonds(tokens, atoms, spi, chains)

    mol = Chem.MolFromSmiles(smiles)
    assert chains[0].source_smiles == smiles
    assert len(tokens) == mol.GetNumAtoms()
    n_edges = int(token_bonds.sum().item()) // 2  # symmetric matrix
    assert n_edges == mol.GetNumBonds()
    assert n_edges < len(tokens) * (len(tokens) - 1) // 2  # not a clique


def test_mau_source_stereo_survives_reference_conformer_recovery():
    """RGI must repair the MAU seed-0 reference that inverts one E/Z bond."""
    from rgi_toolkit._mol_build import (
        _expected_stereo,
        _stereo_mismatch_counts,
        ff_relax,
    )
    from rgi_toolkit.esmfold2.adapter import ESMFold2Adapter

    spi = StructurePredictionInput(
        sequences=[
            LigandInput(id="B", smiles=MAU_SMILES, conformer_restraints=True)
        ]
    )
    features, chains = ESMFold2InputBuilder().prepare_input(spi, seed=0)
    ligand = next(ESMFold2Adapter(features, chains).iter_ligand_confs())
    expected = _expected_stereo(ligand.stereo_mol, ligand.conf_coords)
    assert _stereo_mismatch_counts(ligand.stereo_mol, ligand.conf_coords, expected) == (
        0,
        1,
    )

    repaired = ff_relax(
        ligand.mol,
        ligand.conf_coords,
        stereo_mol=ligand.stereo_mol,
    )
    assert repaired is not None
    assert _stereo_mismatch_counts(ligand.stereo_mol, repaired, expected) == (0, 0)
