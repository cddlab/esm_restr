from esm.models.esmfold2.prepare_input import _smiles_atom_names


def test_smiles_atom_names_preserve_existing_scheme_when_it_fits():
    """Short canonical names retain the historical global-rank scheme."""
    names = _smiles_atom_names(["C", "Cl", "O"], [2, 0, 1])

    assert names == ["C3", "CL1", "O2"]


def test_smiles_atom_names_reindex_by_element_on_overflow():
    """A CL100 overflow switches to unique element-local names."""
    symbols = ["C"] * 99 + ["Cl"] + ["H"] * 80
    names = _smiles_atom_names(symbols, list(range(len(symbols))))

    assert names[99] == "CL1"
    assert len(names) == len(set(names))
    assert all(len(name) <= 4 for name in names)
