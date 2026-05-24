"""
tests/test_structure_3d.py
--------------------------
Tests for ``coordchem.viz.molecule3D``.

Run with:
    python -m pytest tests/test_structure_3d.py -v
"""

import math

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402

from coordchem.complex import Complex  # noqa: E402
from coordchem.parser import KNOWN_LIGANDS, parse_formula  # noqa: E402
from coordchem.viz.molecule3D import (  # noqa: E402
    _cp_ring_atom_groups,
    _cp_ring_centers,
    bidentate_site_pair_indices,
    build_complex_3d,
    build_ligand_3d,
    capped_octahedral_positions,
    dodecahedral_positions,
    find_donor_atom,
    geometry_positions,
    octahedral_positions,
    parse_complex_input,
    pentagonal_bipyramidal_positions,
    view_complex_3d,
)
from coordchem.viz.ligand_data import LIGAND_SMILES  # noqa: E402


class TestGeometryPositions:
    def test_octahedral_returns_six_axes(self):
        pos = octahedral_positions(distance=2.0)
        assert len(pos) == 6
        # Sites are 2 Å away from origin along the axes
        for x, y, z in pos:
            assert pytest_approx(x ** 2 + y ** 2 + z ** 2) == 4.0

    def test_geometry_positions_octahedral(self):
        pos = geometry_positions("octahedral", 6)
        assert len(pos) == 6

    def test_geometry_positions_tetrahedral(self):
        pos = geometry_positions("tetrahedral", 4)
        assert len(pos) == 4

    def test_geometry_positions_unknown_falls_back(self):
        # Unknown geometry but n=6 should still give 6 sites
        pos = geometry_positions("unknown geometry", 6)
        assert len(pos) == 6

    def test_geometry_positions_square_antiprismatic(self):
        pos = geometry_positions("square antiprismatic or dodecahedral", 8)

        assert len(pos) == 8
        assert all(any(abs(component) > 0 for component in site) for site in pos)

    def test_dodecahedral_positions_are_not_flat_octagon_or_antiprism(self):
        pos = dodecahedral_positions(distance=2.0)

        assert len(pos) == 8
        assert all(
            math.dist((0.0, 0.0, 0.0), site) == pytest.approx(2.0)
            for site in pos
        )
        assert len({round(site[2], 2) for site in pos}) == 4

    def test_geometry_positions_explicit_dodecahedral(self):
        pos = geometry_positions("dodecahedral", 8)

        assert pos == dodecahedral_positions()

    def test_short_bidentate_dodecahedral_pairs_are_adjacent(self):
        pairs = bidentate_site_pair_indices("dodecahedral", 8)
        pos = dodecahedral_positions(distance=2.0)
        paired_distances = [math.dist(pos[a], pos[b]) for a, b in pairs]

        assert pairs == [(0, 1), (2, 3), (4, 5), (6, 7)]
        assert all(distance < 2.6 for distance in paired_distances)

    def test_short_bidentate_square_planar_pairs_are_cis(self):
        pairs = bidentate_site_pair_indices("square planar", 4)
        assert pairs == [(0, 2), (1, 3)]

    def test_short_bidentate_pentagonal_bipyramidal_pairs_are_equatorial(self):
        pairs = bidentate_site_pair_indices("pentagonal bipyramidal", 7)
        pos = pentagonal_bipyramidal_positions(distance=2.0)
        paired_distances = [math.dist(pos[a], pos[b]) for a, b in pairs[:5]]

        assert pairs[:5] == [(0, 1), (2, 3), (3, 4), (4, 0), (1, 2)]
        assert all(distance == pytest.approx(2.35, abs=0.01) for distance in paired_distances)

    def test_short_bidentate_cn7_ambiguous_label_uses_pentagonal_pairs(self):
        geometry = "pentagonal bipyramidal or capped octahedral"
        pairs = bidentate_site_pair_indices(geometry, 7)
        pos = geometry_positions(geometry, 7)
        paired_distances = [math.dist(pos[a], pos[b]) for a, b in pairs[:2]]

        assert pos == pentagonal_bipyramidal_positions()
        assert pairs[:2] == [(0, 1), (2, 3)]
        assert all(distance == pytest.approx(2.35, abs=0.01) for distance in paired_distances)

    def test_short_bidentate_capped_octahedral_pairs_are_edges_not_trans(self):
        pairs = bidentate_site_pair_indices("capped octahedral", 7)
        pos = capped_octahedral_positions(distance=2.0)
        paired_distances = [math.dist(pos[a], pos[b]) for a, b in pairs]

        assert pairs[:3] == [(0, 2), (1, 4), (3, 5)]
        assert all(distance < 3.1 for distance in paired_distances)


class TestBuildLigand:
    def test_build_water(self):
        mol = build_ligand_3d("O")
        assert mol.GetNumConformers() == 1
        # H2O has 1 O + 2 H after AddHs
        assert mol.GetNumAtoms() == 3

    def test_build_invalid_smiles(self):
        with pytest.raises(ValueError):
            build_ligand_3d("not_a_smiles_string!!")

    def test_find_donor_in_ammonia(self):
        mol = build_ligand_3d("N")
        idx = find_donor_atom(mol, "N")
        assert mol.GetAtomWithIdx(idx).GetSymbol() == "N"

    def test_find_donor_with_override(self):
        mol = build_ligand_3d("NCCN")  # ethylenediamine
        idx = find_donor_atom(mol, "N", override=0)
        assert idx == 0
        assert mol.GetAtomWithIdx(idx).GetSymbol() == "N"

    def test_find_donor_missing_raises(self):
        mol = build_ligand_3d("N")
        with pytest.raises(ValueError):
            find_donor_atom(mol, "P")


class TestBuildComplex:
    def test_parse_complex_input_accepts_methyl_formula(self):
        parsed = parse_complex_input("[Ti(CH3)4]")

        assert parsed.metal == "Ti"
        assert parsed.ligands == {"CH3": 4}

    def test_hexacyanoferrate_builds(self):
        parsed = parse_formula("[Fe(CN)6]4-")
        mol = build_complex_3d(parsed)

        assert isinstance(mol, Chem.Mol)
        assert mol.GetNumConformers() == 1

        # Metal at index 0 should be Fe at the origin
        assert mol.GetAtomWithIdx(0).GetSymbol() == "Fe"
        conf = mol.GetConformer()
        origin = conf.GetAtomPosition(0)
        assert pytest_approx(origin.x ** 2 + origin.y ** 2 + origin.z ** 2) == 0.0

        # 6 dative bonds from the metal
        metal_atom = mol.GetAtomWithIdx(0)
        dative_bonds = [
            b for b in metal_atom.GetBonds()
            if b.GetBondType() == Chem.BondType.DATIVE
        ]
        assert len(dative_bonds) == 6

    def test_complex_class_draw_3d_html(self):
        py3Dmol = pytest.importorskip("py3Dmol")  # noqa: F841

        c = Complex.from_formula("[Fe(CN)6]4-")
        html = c.draw_3d_html(width=300, height=300)

        assert isinstance(html, str)
        assert len(html) > 0

    def test_complex_class_build_3d(self):
        c = Complex.from_formula("[Fe(CN)6]4-")
        mol = c.build_3d()

        assert mol.GetNumConformers() >= 1
        assert mol.GetAtomWithIdx(0).GetSymbol() == "Fe"

    def test_tetrahedral_complex_has_four_sites(self):
        parsed = parse_formula("[Zn(NH3)4]2+")
        mol = build_complex_3d(parsed)

        metal_atom = mol.GetAtomWithIdx(0)
        dative_bonds = [
            b for b in metal_atom.GetBonds()
            if b.GetBondType() == Chem.BondType.DATIVE
        ]
        assert len(dative_bonds) == 4

    @pytest.mark.parametrize(
        ("formula", "expected_donor_symbols"),
        [
            ("[TaF8]3-", ["F"] * 8),
            ("[Zn(ox)4]6-", ["O"] * 8),
            ("[Zr(ox)2F4]4-", ["O"] * 4 + ["F"] * 4),
        ],
    )
    def test_cn8_3d_handles_monodentate_bidentate_and_mixed_ligands(
        self,
        formula,
        expected_donor_symbols,
    ):
        parsed = parse_formula(formula)
        mol = build_complex_3d(parsed)

        assert parsed.coordination_number == 8
        assert _metal_donor_symbols(mol) == expected_donor_symbols

    def test_methyl_complex_builds_four_ch3_ligands(self):
        parsed = parse_formula("[Ti(CH3)4]")
        mol = build_complex_3d(parsed)

        donor_symbols = _metal_donor_symbols(mol)
        methyl_carbons = [
            atom
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "C"
        ]

        assert parsed.ligands == {"CH3": 4}
        assert donor_symbols == ["C"] * 4
        assert len(methyl_carbons) == 4
        assert all(
            sum(
                1 for neighbor in atom.GetNeighbors()
                if neighbor.GetSymbol() == "H"
            ) == 3
            for atom in methyl_carbons
        )

    def test_methyl_3d_uses_realistic_tetrahedral_ch_geometry(self):
        mol = build_complex_3d(parse_formula("[Ti(CH3)4]"))

        ch_lengths = _bond_lengths_for_symbols(mol, {"C", "H"})
        hch_angles = _center_angles_matching_neighbors(
            mol,
            center_symbol="C",
            neighbor_symbols=("H", "H"),
        )
        metal_c_h_angles = _center_angles_matching_neighbors(
            mol,
            center_symbol="C",
            neighbor_symbols=("Ti", "H"),
        )

        assert len(ch_lengths) == 12
        assert all(length == pytest.approx(1.09, abs=0.03) for length in ch_lengths)
        assert len(hch_angles) == 12
        assert len(metal_c_h_angles) == 12
        assert all(angle == pytest.approx(109.47, abs=0.5) for angle in hch_angles)
        assert all(
            angle == pytest.approx(109.47, abs=0.5)
            for angle in metal_c_h_angles
        )

    def test_ammine_3d_uses_realistic_tetrahedral_nh_geometry(self):
        mol = build_complex_3d(parse_formula("[Co(NH3)6]3+"))

        nh_lengths = _bond_lengths_for_symbols(mol, {"N", "H"})
        hnh_angles = _center_angles_matching_neighbors(
            mol,
            center_symbol="N",
            neighbor_symbols=("H", "H"),
        )
        metal_n_h_angles = _center_angles_matching_neighbors(
            mol,
            center_symbol="N",
            neighbor_symbols=("Co", "H"),
        )

        assert len(nh_lengths) == 18
        assert all(length == pytest.approx(1.01, abs=0.03) for length in nh_lengths)
        assert len(hnh_angles) == 18
        assert len(metal_n_h_angles) == 18
        assert all(angle == pytest.approx(109.47, abs=0.5) for angle in hnh_angles)
        assert all(
            angle == pytest.approx(109.47, abs=0.5)
            for angle in metal_n_h_angles
        )

    def test_pyridine_3d_uses_realistic_aromatic_ring_lengths(self):
        mol = build_complex_3d(parse_formula("[Co(py)6]3+"))

        cc_lengths = _bond_lengths_for_symbols(mol, {"C"})
        cn_lengths = _bond_lengths_for_symbols(mol, {"C", "N"})
        ch_lengths = _bond_lengths_for_symbols(mol, {"C", "H"})

        assert len(cc_lengths) == 24
        assert len(cn_lengths) == 12
        assert len(ch_lengths) == 30
        assert all(length == pytest.approx(1.39, abs=0.03) for length in cc_lengths)
        assert all(length == pytest.approx(1.39, abs=0.03) for length in cn_lengths)
        assert all(length == pytest.approx(1.09, abs=0.03) for length in ch_lengths)

    def test_pyridine_3d_uses_longer_metal_nitrogen_bonds(self):
        mol = build_complex_3d(parse_formula("[Co(py)6]3+"))

        metal_n_lengths = _metal_donor_bond_lengths(mol, donor_symbol="N")

        assert len(metal_n_lengths) == 6
        assert all(
            length == pytest.approx(3.0, abs=0.03)
            for length in metal_n_lengths
        )

    def test_acac_3d_oxygen_donors_have_no_hydrogen_neighbor(self):
        parsed = parse_formula("[Co(acac)3]")
        mol = build_complex_3d(parsed)

        oxygen_donors = [
            mol.GetAtomWithIdx(bond.GetOtherAtomIdx(0))
            for bond in mol.GetAtomWithIdx(0).GetBonds()
            if mol.GetAtomWithIdx(bond.GetOtherAtomIdx(0)).GetSymbol() == "O"
        ]

        assert len(oxygen_donors) == 6
        assert all(
            all(neighbor.GetSymbol() != "H" for neighbor in atom.GetNeighbors())
            for atom in oxygen_donors
        )

    def test_acac_3d_metal_oxygen_carbon_angles_follow_chelate_shape(self):
        parsed = parse_formula("[Co(acac)3]")
        mol = build_complex_3d(parsed)

        angles = _metal_oxygen_carbon_angles(mol)

        assert len(angles) == 6
        assert all(125.0 <= angle <= 145.0 for angle in angles)

    def test_acac_3d_uses_realistic_backbone_bond_lengths(self):
        parsed = parse_formula("[Co(acac)3]")
        mol = build_complex_3d(parsed)

        co_lengths = _bond_lengths_for_symbols(mol, {"C", "O"})
        cc_lengths = _bond_lengths_for_symbols(mol, {"C"})
        ch_lengths = _bond_lengths_for_symbols(mol, {"C", "H"})

        assert len(co_lengths) == 6
        assert len(cc_lengths) == 12
        assert len(ch_lengths) == 21
        assert all(length == pytest.approx(1.28, abs=0.03) for length in co_lengths)
        assert all(1.35 <= length <= 1.55 for length in cc_lengths)
        assert all(length == pytest.approx(1.09, abs=0.03) for length in ch_lengths)

    def test_acac_3d_sp2_carbons_are_trigonal_planar(self):
        parsed = parse_formula("[Co(acac)3]")
        mol = build_complex_3d(parsed)

        sp2_carbons = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "C"
            and atom.GetDegree() == 3
            and sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == "H") <= 1
        ]

        assert len(sp2_carbons) == 9
        for carbon_idx in sp2_carbons:
            neighbors = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(carbon_idx).GetNeighbors()]
            angles = [
                _atom_angle(mol, neighbors[i], carbon_idx, neighbors[j])
                for i in range(len(neighbors))
                for j in range(i + 1, len(neighbors))
            ]

            assert sum(angles) == pytest.approx(360.0, abs=1e-6)
            assert all(105.0 <= angle <= 135.0 for angle in angles)

    def test_acac_3d_methyl_carbons_are_tetrahedral_sp3(self):
        parsed = parse_formula("[Co(acac)3]")
        mol = build_complex_3d(parsed)

        methyl_carbons = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "C"
            and sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == "H") == 3
        ]

        assert len(methyl_carbons) == 6
        for carbon_idx in methyl_carbons:
            hydrogens = [
                neighbor.GetIdx()
                for neighbor in mol.GetAtomWithIdx(carbon_idx).GetNeighbors()
                if neighbor.GetSymbol() == "H"
            ]
            heavy_neighbor = next(
                neighbor.GetIdx()
                for neighbor in mol.GetAtomWithIdx(carbon_idx).GetNeighbors()
                if neighbor.GetSymbol() != "H"
            )

            for hydrogen_idx in hydrogens:
                assert _atom_angle(
                    mol,
                    heavy_neighbor,
                    carbon_idx,
                    hydrogen_idx,
                ) == pytest.approx(109.47, abs=0.5)

            for i in range(len(hydrogens)):
                for j in range(i + 1, len(hydrogens)):
                    assert _atom_angle(
                        mol,
                        hydrogens[i],
                        carbon_idx,
                        hydrogens[j],
                    ) == pytest.approx(109.47, abs=0.5)

    def test_oxalate_3d_uses_realistic_carboxylate_lengths(self):
        parsed = parse_formula("[Fe(ox)3]3-")
        mol = build_complex_3d(parsed)

        carbonyl_lengths = _bond_lengths_for_symbols_and_type(
            mol,
            {"C", "O"},
            Chem.BondType.DOUBLE,
        )
        donor_co_lengths = _bond_lengths_for_symbols_and_type(
            mol,
            {"C", "O"},
            Chem.BondType.SINGLE,
        )
        cc_lengths = _bond_lengths_for_symbols_and_type(
            mol,
            {"C"},
            Chem.BondType.SINGLE,
        )

        assert len(carbonyl_lengths) == 6
        assert len(donor_co_lengths) == 6
        assert len(cc_lengths) == 3
        assert all(length == pytest.approx(1.22, abs=0.03) for length in carbonyl_lengths)
        assert all(length == pytest.approx(1.27, abs=0.03) for length in donor_co_lengths)
        assert all(length == pytest.approx(1.54, abs=0.03) for length in cc_lengths)

    def test_oxalate_3d_carbons_are_trigonal_planar(self):
        parsed = parse_formula("[Fe(ox)3]3-")
        mol = build_complex_3d(parsed)

        oxalate_carbons = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "C"
        ]

        assert len(oxalate_carbons) == 6
        for carbon_idx in oxalate_carbons:
            neighbors = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(carbon_idx).GetNeighbors()]
            angles = [
                _atom_angle(mol, neighbors[i], carbon_idx, neighbors[j])
                for i in range(len(neighbors))
                for j in range(i + 1, len(neighbors))
            ]

            assert sum(angles) == pytest.approx(360.0, abs=1e-6)
            assert all(angle == pytest.approx(120.0, abs=1.0) for angle in angles)

    def test_phen_3d_ligand_atoms_are_in_each_n_metal_n_plane(self):
        parsed = parse_formula("[Fe(phen)3]2+")
        mol = build_complex_3d(parsed)

        deviations = _ligand_plane_deviations(mol, donor_symbol="N")

        assert len(deviations) == 3
        assert all(max(component) < 1e-6 for component in deviations)

    def test_phen_3d_ligands_point_away_from_metal(self):
        parsed = parse_formula("[Fe(phen)3]2+")
        mol = build_complex_3d(parsed)

        outward_dots = _ligand_centroid_outward_dots(mol, donor_symbol="N")

        assert len(outward_dots) == 3
        assert all(dot_value > 0 for dot_value in outward_dots)

    @pytest.mark.parametrize("ligand", ["bipy", "bpy"])
    def test_bipyridine_3d_ligand_atoms_are_in_each_n_metal_n_plane(self, ligand):
        parsed = parse_formula(f"[Fe({ligand})3]2+")
        mol = build_complex_3d(parsed)

        deviations = _ligand_plane_deviations(mol, donor_symbol="N")

        assert len(deviations) == 3
        assert all(max(component) < 1e-6 for component in deviations)

    @pytest.mark.parametrize("ligand", ["bipy", "bpy"])
    def test_bipyridine_3d_ligands_point_away_from_metal(self, ligand):
        parsed = parse_formula(f"[Fe({ligand})3]2+")
        mol = build_complex_3d(parsed)

        outward_dots = _ligand_centroid_outward_dots(mol, donor_symbol="N")

        assert len(outward_dots) == 3
        assert all(dot_value > 0 for dot_value in outward_dots)

    @pytest.mark.parametrize("ligand", ["tpy", "terpy"])
    def test_terpyridine_3d_ligands_are_planar(self, ligand):
        parsed = parse_formula(f"[Ru({ligand})2]2+")
        mol = build_complex_3d(parsed)

        deviations = _tridentate_ligand_plane_deviations(mol, donor_symbol="N")

        assert len(deviations) == 2
        assert all(max(component) < 1e-6 for component in deviations)

    @pytest.mark.parametrize("ligand", ["tpy", "terpy"])
    def test_terpyridine_3d_ligands_do_not_overlap(self, ligand):
        parsed = parse_formula(f"[Ru({ligand})2]2+")
        mol = build_complex_3d(parsed)

        centroids = _ligand_centroids_with_donor_count(
            mol,
            donor_symbol="N",
            donor_count=3,
        )

        assert len(centroids) == 2
        assert math.dist(centroids[0], centroids[1]) > 1.5

    @pytest.mark.parametrize("ligand", ["tpy", "terpy"])
    def test_bis_terpyridine_3d_uses_top_bottom_perpendicular_planes(self, ligand):
        parsed = parse_formula(f"[Ru({ligand})2]2+")
        mol = build_complex_3d(parsed)

        summaries = _ligand_axis_summaries_with_donor_count(
            mol,
            donor_symbol="N",
            donor_count=3,
        )

        assert len(summaries) == 2
        assert summaries[0]["centroid"][2] > 0
        assert summaries[1]["centroid"][2] < 0
        assert summaries[0]["span_y"] == pytest.approx(0.0, abs=1e-6)
        assert summaries[1]["span_x"] == pytest.approx(0.0, abs=1e-6)
        assert summaries[0]["span_x"] > 5.0
        assert summaries[0]["span_z"] > 5.0
        assert summaries[1]["span_y"] > 5.0
        assert summaries[1]["span_z"] > 5.0

    @pytest.mark.parametrize("ligand", ["tpy", "terpy"])
    def test_terpyridine_3d_preserves_uniform_aromatic_ring_lengths(self, ligand):
        parsed = parse_formula(f"[Ru({ligand})2]2+")
        mol = build_complex_3d(parsed)

        aromatic_lengths = _aromatic_heavy_bond_lengths(mol)
        metal_n_lengths = _metal_donor_bond_lengths(mol, donor_symbol="N")

        assert len(aromatic_lengths) == 36
        assert all(length == pytest.approx(1.39, abs=0.01) for length in aromatic_lengths)
        assert len(metal_n_lengths) == 6
        assert sum(1 for length in metal_n_lengths if length > 2.3) == 4

    def test_dmso_hard_metal_uses_oxygen_donor(self):
        parsed = parse_formula("[Fe(dmso)6]3+")
        mol = build_complex_3d(parsed)

        donor_symbols = _metal_donor_symbols(mol)
        assert donor_symbols == ["O"] * 6

    def test_uppercase_dmso_3d_uses_same_donor_logic(self):
        parsed = parse_formula("[Fe(DMSO)6]3+")
        mol = build_complex_3d(parsed)

        assert parsed.ligands == {"dmso": 6}
        assert _metal_donor_symbols(mol) == ["O"] * 6

    def test_dmso_soft_metal_uses_sulfur_donor(self):
        parsed = parse_formula("[Pt(dmso)4]2+")
        mol = build_complex_3d(parsed)

        donor_symbols = _metal_donor_symbols(mol)
        assert donor_symbols == ["S"] * 4

    def test_dmso_3d_sulfur_donor_has_no_hydrogen_neighbor(self):
        parsed = parse_formula("[Pt(dmso)4]2+")
        mol = build_complex_3d(parsed)

        sulfur_donors = [
            mol.GetAtomWithIdx(bond.GetOtherAtomIdx(0))
            for bond in mol.GetAtomWithIdx(0).GetBonds()
            if mol.GetAtomWithIdx(bond.GetOtherAtomIdx(0)).GetSymbol() == "S"
        ]

        assert len(sulfur_donors) == 4
        assert all(
            all(neighbor.GetSymbol() != "H" for neighbor in atom.GetNeighbors())
            for atom in sulfur_donors
        )

    def test_dmso_sulfur_donor_3d_uses_tetrahedral_sulfur(self):
        parsed = parse_formula("[Pt(dmso)4]2+")
        mol = build_complex_3d(parsed)

        sulfur_angles = _center_angles_for_symbol(mol, "S")
        sc_lengths = _bond_lengths_for_symbols(mol, {"S", "C"})
        so_lengths = _bond_lengths_for_symbols(mol, {"S", "O"})

        assert len(sulfur_angles) == 24
        assert all(angle == pytest.approx(109.47, abs=0.5) for angle in sulfur_angles)
        assert len(sc_lengths) == 8
        assert all(length == pytest.approx(1.79, abs=0.03) for length in sc_lengths)
        assert len(so_lengths) == 4
        assert all(length == pytest.approx(1.49, abs=0.03) for length in so_lengths)

    def test_dmso_oxygen_donor_3d_has_bent_metal_oxygen_sulfur_angle(self):
        parsed = parse_formula("[Fe(dmso)6]3+")
        mol = build_complex_3d(parsed)

        metal_oxygen_sulfur_angles = _metal_oxygen_sulfur_angles(mol)
        sulfur_angles = _center_angles_for_symbol(mol, "S")

        assert len(metal_oxygen_sulfur_angles) == 6
        assert all(
            angle == pytest.approx(120.0, abs=0.5)
            for angle in metal_oxygen_sulfur_angles
        )
        assert len(sulfur_angles) == 18
        assert all(angle == pytest.approx(109.47, abs=0.5) for angle in sulfur_angles)

    @pytest.mark.parametrize("ligand", ["SCN", "NCS"])
    def test_thiocyanato_3d_is_linear_with_realistic_lengths(self, ligand):
        parsed = parse_formula(f"[Co({ligand})6]")
        mol = build_complex_3d(parsed)

        carbon_angles = _center_angles_for_symbol(mol, "C")
        cn_lengths = _bond_lengths_for_symbols(mol, {"C", "N"})
        cs_lengths = _bond_lengths_for_symbols(mol, {"C", "S"})

        assert len(carbon_angles) == 6
        assert all(angle == pytest.approx(180.0, abs=0.5) for angle in carbon_angles)
        assert len(cn_lengths) == 6
        assert all(length == pytest.approx(1.16, abs=0.03) for length in cn_lengths)
        assert len(cs_lengths) == 6
        assert all(length == pytest.approx(1.63, abs=0.03) for length in cs_lengths)

    def test_azido_3d_is_linear_and_points_outward(self):
        parsed = parse_formula("[Co(N3)6]")
        mol = build_complex_3d(parsed)

        azide_chains = _metal_bound_linear_chains(mol, donor_symbol="N", chain_symbol="N")
        nn_lengths = _bond_lengths_for_symbols(mol, {"N"})

        assert sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "H") == 0
        assert len(azide_chains) == 6
        assert all(len(chain) == 3 for chain in azide_chains)
        assert all(length == pytest.approx(1.18, abs=0.03) for length in nn_lengths)
        assert all(
            dot_value == pytest.approx(1.0, abs=1e-6)
            for chain in azide_chains
            for dot_value in _chain_outward_dots(mol, chain)
        )

    def test_nitro_3d_uses_trigonal_planar_nitrogen(self):
        parsed = parse_formula("[Co(NO2)6]")
        mol = build_complex_3d(parsed)

        nitrogen_angles = _center_angles_for_symbol(mol, "N")
        no_lengths = _bond_lengths_for_symbols(mol, {"N", "O"})
        ligand_mol = Chem.MolFromSmiles(LIGAND_SMILES["NO2"])

        assert Chem.GetFormalCharge(ligand_mol) == -1
        assert len(nitrogen_angles) == 18
        assert all(angle == pytest.approx(120.0, abs=0.5) for angle in nitrogen_angles)
        assert len(no_lengths) == 12
        assert all(length == pytest.approx(1.24, abs=0.03) for length in no_lengths)

    def test_nitrito_o_3d_uses_bent_o_binding_and_planar_no2(self):
        parsed = parse_formula("[Co(ONO)6]")
        mol = build_complex_3d(parsed)

        nitrogen_angles = _center_angles_for_symbol(mol, "N")
        metal_oxygen_nitrogen_angles = _metal_oxygen_neighbor_angles(
            mol,
            neighbor_symbol="N",
        )
        no_lengths = sorted(_bond_lengths_for_symbols(mol, {"N", "O"}))
        ligand_mol = Chem.MolFromSmiles(LIGAND_SMILES["ONO"])

        assert Chem.GetFormalCharge(ligand_mol) == -1
        assert len(nitrogen_angles) == 6
        assert all(angle == pytest.approx(120.0, abs=0.5) for angle in nitrogen_angles)
        assert len(metal_oxygen_nitrogen_angles) == 6
        assert all(
            angle == pytest.approx(120.0, abs=0.5)
            for angle in metal_oxygen_nitrogen_angles
        )
        assert len(no_lengths) == 12
        assert no_lengths[:6] == pytest.approx([1.22] * 6, abs=0.03)
        assert no_lengths[6:] == pytest.approx([1.30] * 6, abs=0.03)

    def test_nitrosyl_uses_no_plus_linear_convention(self):
        parsed = parse_formula("[Co(NO)6]")
        mol = build_complex_3d(parsed)
        ligand_mol = Chem.MolFromSmiles(LIGAND_SMILES["NO"])

        metal_nitrogen_oxygen_angles = _center_angles_matching_neighbors(
            mol,
            center_symbol="N",
            neighbor_symbols=("Co", "O"),
        )

        assert KNOWN_LIGANDS["NO"][1] == 1
        assert Chem.GetFormalCharge(ligand_mol) == 1
        assert len(metal_nitrogen_oxygen_angles) == 6
        assert all(
            angle == pytest.approx(180.0, abs=0.5)
            for angle in metal_nitrogen_oxygen_angles
        )

    def test_aqua_3d_uses_realistic_hoh_angle(self):
        parsed = parse_formula("[Co(H2O)6]")
        mol = build_complex_3d(parsed)

        hoh_angles = _center_angles_matching_neighbors(
            mol,
            center_symbol="O",
            neighbor_symbols=("H", "H"),
        )
        oh_lengths = _bond_lengths_for_symbols(mol, {"O", "H"})

        assert len(hoh_angles) == 6
        assert all(angle == pytest.approx(104.5, abs=0.5) for angle in hoh_angles)
        assert len(oh_lengths) == 12
        assert all(length == pytest.approx(0.96, abs=0.03) for length in oh_lengths)

    def test_hydroxo_3d_is_bent_not_linear(self):
        parsed = parse_formula("[Co(OH)6]")
        mol = build_complex_3d(parsed)

        metal_oxygen_hydrogen_angles = _center_angles_matching_neighbors(
            mol,
            center_symbol="O",
            neighbor_symbols=("Co", "H"),
        )
        oh_lengths = _bond_lengths_for_symbols(mol, {"O", "H"})

        assert len(metal_oxygen_hydrogen_angles) == 6
        assert all(
            angle == pytest.approx(109.47, abs=0.5)
            for angle in metal_oxygen_hydrogen_angles
        )
        assert len(oh_lengths) == 6
        assert all(length == pytest.approx(0.96, abs=0.03) for length in oh_lengths)

    def test_edta_3d_methylene_hydrogens_are_tetrahedral(self):
        parsed = parse_formula("[Co(EDTA)]")
        mol = build_complex_3d(parsed)

        hch_angles = _carbon_hydrogen_hydrogen_angles(mol)

        assert len(hch_angles) == 6
        assert all(angle == pytest.approx(109.47, abs=0.5) for angle in hch_angles)

    def test_ethylenediamine_3d_uses_realistic_sp3_chain(self):
        mol = build_complex_3d(parse_formula("[Co(en)3]3+"))

        nc_lengths = _bond_lengths_for_symbols(mol, {"N", "C"})
        cc_lengths = _bond_lengths_for_symbols(mol, {"C"})
        nh_lengths = _bond_lengths_for_symbols(mol, {"N", "H"})
        ch_lengths = _bond_lengths_for_symbols(mol, {"C", "H"})
        metal_n_c_angles = _center_angles_matching_neighbors(
            mol,
            center_symbol="N",
            neighbor_symbols=("Co", "C"),
        )
        hch_angles = _carbon_hydrogen_hydrogen_angles(mol)

        assert len(nc_lengths) == 6
        assert len(cc_lengths) == 3
        assert len(nh_lengths) == 12
        assert len(ch_lengths) == 12
        assert all(length == pytest.approx(1.47, abs=0.03) for length in nc_lengths)
        assert all(length == pytest.approx(1.53, abs=0.03) for length in cc_lengths)
        assert all(length == pytest.approx(1.01, abs=0.03) for length in nh_lengths)
        assert all(length == pytest.approx(1.09, abs=0.03) for length in ch_lengths)
        assert len(metal_n_c_angles) == 6
        assert all(
            angle == pytest.approx(109.5, abs=2.0)
            for angle in metal_n_c_angles
        )
        assert all(angle == pytest.approx(109.5, abs=1.0) for angle in hch_angles)

    def test_cp_3d_uses_no_dummy_atom_or_artificial_metal_carbon_bonds(self):
        parsed = parse_formula("[Co(Cp)6]")
        mol = build_complex_3d(parsed)

        assert sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "He") == 0
        assert sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0) == 0
        assert len(
            [
                bond
                for bond in mol.GetAtomWithIdx(0).GetBonds()
                if bond.GetBondType() == Chem.BondType.DATIVE
                and mol.GetAtomWithIdx(bond.GetOtherAtomIdx(0)).GetSymbol() == "C"
            ]
        ) == 0

    def test_cp_3d_uses_realistic_aromatic_ring_lengths(self):
        mol = build_complex_3d(parse_formula("[Co(Cp)6]"))

        cc_lengths = _bond_lengths_for_symbols(mol, {"C"})
        ch_lengths = _bond_lengths_for_symbols(mol, {"C", "H"})

        assert len(cc_lengths) == 30
        assert len(ch_lengths) == 30
        assert all(length == pytest.approx(1.40, abs=0.03) for length in cc_lengths)
        assert all(length == pytest.approx(1.09, abs=0.03) for length in ch_lengths)

    def test_cp_3d_uses_single_ring_bonds_with_dashed_delocalization(self):
        mol = build_complex_3d(parse_formula("[Co(Cp)6]"))

        cp_cc_bonds = [
            bond
            for bond in mol.GetBonds()
            if {
                mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol(),
                mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol(),
            } == {"C"}
        ]

        assert len(cp_cc_bonds) == 30
        assert all(bond.GetBondType() == Chem.BondType.SINGLE for bond in cp_cc_bonds)
        assert not any(
            atom.GetIsAromatic()
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "C"
        )

    @pytest.mark.parametrize(
        ("formula", "geometry", "expected_pairs"),
        [
            ("[Fe(Cp)2]", None, 1),
            ("[Ti(Cp)4]", "square planar", 2),
            ("[Co(Cp)6]", "octahedral", 3),
        ],
    )
    def test_cp_3d_places_opposite_rings_staggered(
        self,
        formula,
        geometry,
        expected_pairs,
    ):
        mol = build_complex_3d(parse_formula(formula), geometry=geometry)
        phase_differences = _cp_antipodal_phase_differences(mol)

        assert len(phase_differences) == expected_pairs
        assert all(
            phase_difference == pytest.approx(36.0, abs=0.5)
            for phase_difference in phase_differences
        )

    def test_cp_3d_uses_longer_metal_ring_center_distance(self):
        mol = build_complex_3d(parse_formula("[Co(Cp)6]"))

        ring_center_distances = [
            math.dist((0.0, 0.0, 0.0), center)
            for center in _cp_ring_centers(mol)
        ]

        assert len(ring_center_distances) == 6
        assert all(
            distance == pytest.approx(3.0, abs=0.03)
            for distance in ring_center_distances
        )

    def test_cp_3d_view_draws_centroid_lines_and_delocalization_rings(self):
        pytest.importorskip("py3Dmol")

        view = view_complex_3d("[Co(Cp)6]")
        html = view._make_html()

        assert "addCylinder" in html
        assert html.count("addCylinder") == 132

    def test_hydride_complex_builds_h_donors(self):
        parsed = parse_formula("[FeH6]4-")
        mol = build_complex_3d(parsed)

        donor_symbols = _metal_donor_symbols(mol)
        hydride_donors = [
            mol.GetAtomWithIdx(bond.GetOtherAtomIdx(0))
            for bond in mol.GetAtomWithIdx(0).GetBonds()
            if mol.GetAtomWithIdx(bond.GetOtherAtomIdx(0)).GetSymbol() == "H"
        ]

        assert parsed.ligands == {"H": 6}
        assert donor_symbols == ["H"] * 6
        assert len(hydride_donors) == 6
        assert all(atom.GetFormalCharge() == -1 for atom in hydride_donors)

    def test_pph3_3d_uses_tetrahedral_phosphorus_and_outward_phenyls(self):
        parsed = parse_formula("[Pt(PPh3)4]2+")
        mol = build_complex_3d(parsed)

        p_centered_angles = _phosphorus_substituent_angles(mol)
        pc_lengths = _bond_lengths_for_symbols(mol, {"P", "C"})
        aromatic_lengths = _aromatic_heavy_bond_lengths(mol)
        outward_dots = _phosphorus_carbon_outward_dots(mol)

        assert len(p_centered_angles) == 24
        assert all(angle == pytest.approx(109.47, abs=0.5) for angle in p_centered_angles)
        assert len(pc_lengths) == 12
        assert all(length == pytest.approx(1.83, abs=0.03) for length in pc_lengths)
        assert len(aromatic_lengths) == 72
        assert all(length == pytest.approx(1.39, abs=0.01) for length in aromatic_lengths)
        assert all(dot_value == pytest.approx(1.0 / 3.0, abs=0.01) for dot_value in outward_dots)

    @pytest.mark.parametrize(
        ("ligand", "expected_angle_count", "expected_cc_count"),
        [
            ("PMe3", 144, 0),
            ("PEt3", 252, 18),
        ],
    )
    def test_trialkylphosphines_3d_use_tetrahedral_p_and_c_geometry(
        self,
        ligand,
        expected_angle_count,
        expected_cc_count,
    ):
        parsed = parse_formula(f"[Co({ligand})6]")
        mol = build_complex_3d(parsed)

        sp3_angles = _phosphine_sp3_angles(mol)
        pc_lengths = _bond_lengths_for_symbols(mol, {"P", "C"})
        cc_lengths = _bond_lengths_for_symbols(mol, {"C"})

        assert len(sp3_angles) == expected_angle_count
        assert all(angle == pytest.approx(109.47, abs=0.5) for angle in sp3_angles)
        assert len(pc_lengths) == 18
        assert all(length == pytest.approx(1.83, abs=0.03) for length in pc_lengths)
        assert len(cc_lengths) == expected_cc_count
        assert all(length == pytest.approx(1.53, abs=0.03) for length in cc_lengths)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metal_donor_symbols(mol):
    metal_atom = mol.GetAtomWithIdx(0)
    return [
        mol.GetAtomWithIdx(bond.GetOtherAtomIdx(0)).GetSymbol()
        for bond in metal_atom.GetBonds()
        if bond.GetBondType() == Chem.BondType.DATIVE
    ]


def _metal_oxygen_carbon_angles(mol):
    conf = mol.GetConformer()
    metal_pos = conf.GetAtomPosition(0)
    angles = []

    for bond in mol.GetAtomWithIdx(0).GetBonds():
        donor_idx = bond.GetOtherAtomIdx(0)
        donor_atom = mol.GetAtomWithIdx(donor_idx)
        if donor_atom.GetSymbol() != "O":
            continue

        carbon_idx = next(
            neighbor.GetIdx()
            for neighbor in donor_atom.GetNeighbors()
            if neighbor.GetSymbol() == "C"
        )
        donor_pos = conf.GetAtomPosition(donor_idx)
        carbon_pos = conf.GetAtomPosition(carbon_idx)

        metal_vec = (
            metal_pos.x - donor_pos.x,
            metal_pos.y - donor_pos.y,
            metal_pos.z - donor_pos.z,
        )
        carbon_vec = (
            carbon_pos.x - donor_pos.x,
            carbon_pos.y - donor_pos.y,
            carbon_pos.z - donor_pos.z,
        )
        numerator = sum(a * b for a, b in zip(metal_vec, carbon_vec))
        denominator = math.sqrt(sum(a * a for a in metal_vec)) * math.sqrt(
            sum(a * a for a in carbon_vec)
        )
        cos_angle = max(-1.0, min(1.0, numerator / denominator))
        angles.append(math.degrees(math.acos(cos_angle)))

    return angles


def _metal_oxygen_sulfur_angles(mol):
    return _metal_oxygen_neighbor_angles(mol, neighbor_symbol="S")


def _metal_oxygen_neighbor_angles(mol, neighbor_symbol):
    conf = mol.GetConformer()
    metal_pos = conf.GetAtomPosition(0)
    angles = []

    for bond in mol.GetAtomWithIdx(0).GetBonds():
        donor_idx = bond.GetOtherAtomIdx(0)
        donor_atom = mol.GetAtomWithIdx(donor_idx)
        if donor_atom.GetSymbol() != "O":
            continue

        neighbor_idx = next(
            neighbor.GetIdx()
            for neighbor in donor_atom.GetNeighbors()
            if neighbor.GetSymbol() == neighbor_symbol
        )
        donor_pos = conf.GetAtomPosition(donor_idx)
        neighbor_pos = conf.GetAtomPosition(neighbor_idx)

        metal_vec = (
            metal_pos.x - donor_pos.x,
            metal_pos.y - donor_pos.y,
            metal_pos.z - donor_pos.z,
        )
        neighbor_vec = (
            neighbor_pos.x - donor_pos.x,
            neighbor_pos.y - donor_pos.y,
            neighbor_pos.z - donor_pos.z,
        )
        numerator = sum(a * b for a, b in zip(metal_vec, neighbor_vec))
        denominator = math.sqrt(sum(a * a for a in metal_vec)) * math.sqrt(
            sum(a * a for a in neighbor_vec)
        )
        cos_angle = max(-1.0, min(1.0, numerator / denominator))
        angles.append(math.degrees(math.acos(cos_angle)))

    return angles


def _center_angles_for_symbol(mol, symbol):
    angles = []

    for atom in mol.GetAtoms():
        if atom.GetIdx() == 0 or atom.GetSymbol() != symbol:
            continue

        neighbor_indices = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        if len(neighbor_indices) < 2:
            continue

        for i in range(len(neighbor_indices)):
            for j in range(i + 1, len(neighbor_indices)):
                angles.append(
                    _atom_angle(
                        mol,
                        neighbor_indices[i],
                        atom.GetIdx(),
                        neighbor_indices[j],
                    )
                )

    return angles


def _center_angles_matching_neighbors(mol, center_symbol, neighbor_symbols):
    expected = sorted(neighbor_symbols)
    angles = []

    for atom in mol.GetAtoms():
        if atom.GetIdx() == 0 or atom.GetSymbol() != center_symbol:
            continue

        neighbor_indices = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        for i in range(len(neighbor_indices)):
            for j in range(i + 1, len(neighbor_indices)):
                actual = sorted(
                    [
                        mol.GetAtomWithIdx(neighbor_indices[i]).GetSymbol(),
                        mol.GetAtomWithIdx(neighbor_indices[j]).GetSymbol(),
                    ]
                )
                if actual != expected:
                    continue

                angles.append(
                    _atom_angle(
                        mol,
                        neighbor_indices[i],
                        atom.GetIdx(),
                        neighbor_indices[j],
                    )
                )

    return angles


def _carbon_hydrogen_hydrogen_angles(mol):
    angles = []

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue

        hydrogen_indices = [
            neighbor.GetIdx()
            for neighbor in atom.GetNeighbors()
            if neighbor.GetSymbol() == "H"
        ]
        if len(hydrogen_indices) != 2:
            continue

        angles.append(
            _atom_angle(
                mol,
                hydrogen_indices[0],
                atom.GetIdx(),
                hydrogen_indices[1],
            )
        )

    return angles


def _metal_bound_linear_chains(mol, donor_symbol, chain_symbol):
    metal_atom = mol.GetAtomWithIdx(0)
    chains = []

    for bond in metal_atom.GetBonds():
        donor_idx = bond.GetOtherAtomIdx(0)
        if mol.GetAtomWithIdx(donor_idx).GetSymbol() != donor_symbol:
            continue

        chain = [donor_idx]
        previous_idx = 0
        current_idx = donor_idx
        while True:
            next_idx = next(
                (
                    neighbor.GetIdx()
                    for neighbor in mol.GetAtomWithIdx(current_idx).GetNeighbors()
                    if neighbor.GetSymbol() == chain_symbol
                    and neighbor.GetIdx() != previous_idx
                ),
                None,
            )
            if next_idx is None:
                break

            chain.append(next_idx)
            previous_idx, current_idx = current_idx, next_idx

        chains.append(chain)

    return chains


def _chain_outward_dots(mol, chain):
    conf = mol.GetConformer()
    donor_pos = conf.GetAtomPosition(chain[0])
    donor_point = (donor_pos.x, donor_pos.y, donor_pos.z)
    outward = _unit_tuple(donor_point)
    dots = []

    for begin_idx, end_idx in zip(chain, chain[1:]):
        begin_pos = conf.GetAtomPosition(begin_idx)
        end_pos = conf.GetAtomPosition(end_idx)
        bond_direction = _unit_tuple(
            (
                end_pos.x - begin_pos.x,
                end_pos.y - begin_pos.y,
                end_pos.z - begin_pos.z,
            )
        )
        dots.append(_dot_tuple(bond_direction, outward))

    return dots


def _bond_lengths_for_symbols(mol, symbols):
    conf = mol.GetConformer()
    lengths = []

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_symbols = {
            mol.GetAtomWithIdx(begin_idx).GetSymbol(),
            mol.GetAtomWithIdx(end_idx).GetSymbol(),
        }
        if bond_symbols != symbols:
            continue

        begin_pos = conf.GetAtomPosition(begin_idx)
        end_pos = conf.GetAtomPosition(end_idx)
        lengths.append(
            math.dist(
                (begin_pos.x, begin_pos.y, begin_pos.z),
                (end_pos.x, end_pos.y, end_pos.z),
            )
        )

    return lengths


def _bond_lengths_for_symbols_and_type(mol, symbols, bond_type):
    conf = mol.GetConformer()
    lengths = []

    for bond in mol.GetBonds():
        if bond.GetBondType() != bond_type:
            continue

        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_symbols = {
            mol.GetAtomWithIdx(begin_idx).GetSymbol(),
            mol.GetAtomWithIdx(end_idx).GetSymbol(),
        }
        if bond_symbols != symbols:
            continue

        begin_pos = conf.GetAtomPosition(begin_idx)
        end_pos = conf.GetAtomPosition(end_idx)
        lengths.append(
            math.dist(
                (begin_pos.x, begin_pos.y, begin_pos.z),
                (end_pos.x, end_pos.y, end_pos.z),
            )
        )

    return lengths


def _aromatic_heavy_bond_lengths(mol):
    conf = mol.GetConformer()
    lengths = []

    for bond in mol.GetBonds():
        if not bond.GetIsAromatic():
            continue

        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if "H" in {
            mol.GetAtomWithIdx(begin_idx).GetSymbol(),
            mol.GetAtomWithIdx(end_idx).GetSymbol(),
        }:
            continue

        begin_pos = conf.GetAtomPosition(begin_idx)
        end_pos = conf.GetAtomPosition(end_idx)
        lengths.append(
            math.dist(
                (begin_pos.x, begin_pos.y, begin_pos.z),
                (end_pos.x, end_pos.y, end_pos.z),
            )
        )

    return lengths


def _metal_donor_bond_lengths(mol, donor_symbol):
    conf = mol.GetConformer()
    metal_pos = conf.GetAtomPosition(0)
    lengths = []

    for bond in mol.GetAtomWithIdx(0).GetBonds():
        donor_idx = bond.GetOtherAtomIdx(0)
        if mol.GetAtomWithIdx(donor_idx).GetSymbol() != donor_symbol:
            continue

        donor_pos = conf.GetAtomPosition(donor_idx)
        lengths.append(
            math.dist(
                (metal_pos.x, metal_pos.y, metal_pos.z),
                (donor_pos.x, donor_pos.y, donor_pos.z),
            )
        )

    return lengths


def _phosphorus_substituent_angles(mol):
    angles = []

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "P":
            continue

        neighbor_indices = [
            neighbor.GetIdx()
            for neighbor in atom.GetNeighbors()
            if neighbor.GetSymbol() in {"C", mol.GetAtomWithIdx(0).GetSymbol()}
        ]
        for i in range(len(neighbor_indices)):
            for j in range(i + 1, len(neighbor_indices)):
                angles.append(
                    _atom_angle(
                        mol,
                        neighbor_indices[i],
                        atom.GetIdx(),
                        neighbor_indices[j],
                    )
                )

    return angles


def _phosphorus_carbon_outward_dots(mol):
    conf = mol.GetConformer()
    metal_pos = conf.GetAtomPosition(0)
    metal = (metal_pos.x, metal_pos.y, metal_pos.z)
    dots = []

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "P":
            continue

        p_pos = conf.GetAtomPosition(atom.GetIdx())
        p_point = (p_pos.x, p_pos.y, p_pos.z)
        outward = _unit_tuple(_sub_tuple(p_point, metal))

        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() != "C":
                continue

            c_pos = conf.GetAtomPosition(neighbor.GetIdx())
            pc_vector = _unit_tuple(
                _sub_tuple((c_pos.x, c_pos.y, c_pos.z), p_point)
            )
            dots.append(_dot_tuple(pc_vector, outward))

    return dots


def _phosphine_sp3_angles(mol):
    angles = []

    for atom in mol.GetAtoms():
        if atom.GetIdx() == 0 or atom.GetSymbol() not in {"P", "C"}:
            continue

        neighbor_indices = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        if len(neighbor_indices) < 2:
            continue

        for i in range(len(neighbor_indices)):
            for j in range(i + 1, len(neighbor_indices)):
                angles.append(
                    _atom_angle(
                        mol,
                        neighbor_indices[i],
                        atom.GetIdx(),
                        neighbor_indices[j],
                    )
                )

    return angles


def _atom_angle(mol, begin_idx, center_idx, end_idx):
    conf = mol.GetConformer()
    begin_pos = conf.GetAtomPosition(begin_idx)
    center_pos = conf.GetAtomPosition(center_idx)
    end_pos = conf.GetAtomPosition(end_idx)

    begin_vec = (
        begin_pos.x - center_pos.x,
        begin_pos.y - center_pos.y,
        begin_pos.z - center_pos.z,
    )
    end_vec = (
        end_pos.x - center_pos.x,
        end_pos.y - center_pos.y,
        end_pos.z - center_pos.z,
    )
    numerator = sum(a * b for a, b in zip(begin_vec, end_vec))
    denominator = math.sqrt(sum(a * a for a in begin_vec)) * math.sqrt(
        sum(a * a for a in end_vec)
    )
    cos_angle = max(-1.0, min(1.0, numerator / denominator))
    return math.degrees(math.acos(cos_angle))


def _ligand_plane_deviations(mol, donor_symbol):
    conf = mol.GetConformer()
    metal_pos = conf.GetAtomPosition(0)
    metal = (metal_pos.x, metal_pos.y, metal_pos.z)

    deviations = []
    for component, donor_indices in _components_with_two_donors(mol, donor_symbol):
        donor_positions = []
        for donor_idx in donor_indices:
            pos = conf.GetAtomPosition(donor_idx)
            donor_positions.append((pos.x, pos.y, pos.z))

        normal = _unit_tuple(
            _cross_tuple(
                _sub_tuple(donor_positions[0], metal),
                _sub_tuple(donor_positions[1], metal),
            )
        )
        component_deviations = []
        for idx in component:
            pos = conf.GetAtomPosition(idx)
            point = (pos.x, pos.y, pos.z)
            component_deviations.append(abs(_dot_tuple(_sub_tuple(point, metal), normal)))
        deviations.append(component_deviations)

    return deviations


def _ligand_centroid_outward_dots(mol, donor_symbol):
    conf = mol.GetConformer()
    dots = []

    for component, donor_indices in _components_with_two_donors(mol, donor_symbol):
        donor_positions = []
        for donor_idx in donor_indices:
            pos = conf.GetAtomPosition(donor_idx)
            donor_positions.append((pos.x, pos.y, pos.z))
        donor_mid = (
            sum(pos[0] for pos in donor_positions) / 2,
            sum(pos[1] for pos in donor_positions) / 2,
            sum(pos[2] for pos in donor_positions) / 2,
        )

        centroid = (
            sum(conf.GetAtomPosition(idx).x for idx in component) / len(component),
            sum(conf.GetAtomPosition(idx).y for idx in component) / len(component),
            sum(conf.GetAtomPosition(idx).z for idx in component) / len(component),
        )
        dots.append(_dot_tuple(_sub_tuple(centroid, donor_mid), donor_mid))

    return dots


def _tridentate_ligand_plane_deviations(mol, donor_symbol):
    conf = mol.GetConformer()
    deviations = []

    for component, donor_indices in _components_with_donor_count(mol, donor_symbol, 3):
        donor_positions = []
        for donor_idx in donor_indices:
            pos = conf.GetAtomPosition(donor_idx)
            donor_positions.append((pos.x, pos.y, pos.z))

        normal = _unit_tuple(
            _cross_tuple(
                _sub_tuple(donor_positions[1], donor_positions[0]),
                _sub_tuple(donor_positions[2], donor_positions[0]),
            )
        )
        component_deviations = []
        for idx in component:
            pos = conf.GetAtomPosition(idx)
            point = (pos.x, pos.y, pos.z)
            component_deviations.append(
                abs(_dot_tuple(_sub_tuple(point, donor_positions[0]), normal))
            )
        deviations.append(component_deviations)

    return deviations


def _ligand_centroids_with_donor_count(mol, donor_symbol, donor_count):
    conf = mol.GetConformer()
    centroids = []

    for component, _ in _components_with_donor_count(mol, donor_symbol, donor_count):
        centroids.append(
            (
                sum(conf.GetAtomPosition(idx).x for idx in component) / len(component),
                sum(conf.GetAtomPosition(idx).y for idx in component) / len(component),
                sum(conf.GetAtomPosition(idx).z for idx in component) / len(component),
            )
        )

    return centroids


def _ligand_axis_summaries_with_donor_count(mol, donor_symbol, donor_count):
    conf = mol.GetConformer()
    summaries = []

    for component, _ in _components_with_donor_count(mol, donor_symbol, donor_count):
        points = [conf.GetAtomPosition(idx) for idx in component]
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        zs = [point.z for point in points]
        summaries.append(
            {
                "centroid": (
                    sum(xs) / len(xs),
                    sum(ys) / len(ys),
                    sum(zs) / len(zs),
                ),
                "span_x": max(xs) - min(xs),
                "span_y": max(ys) - min(ys),
                "span_z": max(zs) - min(zs),
            }
        )

    return summaries


def _components_with_two_donors(mol, donor_symbol):
    yield from _components_with_donor_count(mol, donor_symbol, 2)


def _components_with_donor_count(mol, donor_symbol, donor_count):
    adjacency = {idx: set() for idx in range(mol.GetNumAtoms()) if idx != 0}
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if 0 in {begin_idx, end_idx}:
            continue
        adjacency[begin_idx].add(end_idx)
        adjacency[end_idx].add(begin_idx)

    components = []
    seen = set()
    for start in adjacency:
        if start in seen:
            continue
        stack = [start]
        component = []
        seen.add(start)
        while stack:
            idx = stack.pop()
            component.append(idx)
            for neighbor in adjacency[idx]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)

    metal_atom = mol.GetAtomWithIdx(0)
    for component in components:
        donor_indices = [
            bond.GetOtherAtomIdx(0)
            for bond in metal_atom.GetBonds()
            if bond.GetOtherAtomIdx(0) in component
            and mol.GetAtomWithIdx(bond.GetOtherAtomIdx(0)).GetSymbol() == donor_symbol
        ]
        if len(donor_indices) != donor_count:
            continue

        yield component, donor_indices


def _sub_tuple(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot_tuple(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross_tuple(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _unit_tuple(v):
    length = math.sqrt(sum(component * component for component in v))
    return tuple(component / length for component in v)


def _cp_antipodal_phase_differences(mol):
    conf = mol.GetConformer()
    rings = _cp_ring_atom_groups(mol)
    centers = _cp_ring_centers(mol)
    phase_differences = []

    for first_idx, first_center in enumerate(centers):
        for second_idx, second_center in enumerate(centers[first_idx + 1:], first_idx + 1):
            antipodal_alignment = _dot_tuple(first_center, second_center) / (
                math.dist((0.0, 0.0, 0.0), first_center)
                * math.dist((0.0, 0.0, 0.0), second_center)
            )
            if antipodal_alignment > -0.95:
                continue

            axis = _unit_tuple(_sub_tuple(first_center, second_center))
            reference = (
                (0.0, 1.0, 0.0)
                if abs(axis[1]) < 0.9
                else (0.0, 0.0, 1.0)
            )
            u = _unit_tuple(_cross_tuple(axis, reference))
            v = _unit_tuple(_cross_tuple(axis, u))

            phases = []
            for ring, center in (
                (rings[first_idx], first_center),
                (rings[second_idx], second_center),
            ):
                atom_pos = conf.GetAtomPosition(ring[0])
                radial = _sub_tuple((atom_pos.x, atom_pos.y, atom_pos.z), center)
                phases.append(
                    math.degrees(
                        math.atan2(_dot_tuple(radial, v), _dot_tuple(radial, u))
                    )
                    % 72.0
                )

            phase_difference = abs((phases[1] - phases[0]) % 72.0)
            phase_differences.append(min(phase_difference, 72.0 - phase_difference))

    return phase_differences


def pytest_approx(value, rel=1e-6, abs_=1e-6):
    return pytest.approx(value, rel=rel, abs=abs_)
