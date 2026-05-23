
from typing import Iterable, Optional, Sequence, Tuple
import math
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor

from coordchem.parser import ParsedComplex, parse_formula, KNOWN_LIGANDS
from coordchem.viz.ligand_data import (
    LIGAND_DONOR_INDEX_OVERRIDES,
    LIGAND_SMILES,
    donor_index_overrides_for_ligand,
    is_short_bidentate_ligand,
)
from coordchem.name import parse_name

Position = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Geometry placeholders
# ---------------------------------------------------------------------------

def octahedral_positions(distance: float = 2.0) -> list[Position]:
    """Return six orthogonal positions on the +/- x, y, z axes."""
    return [
        ( distance,  0.0,       0.0),
        (-distance,  0.0,       0.0),
        ( 0.0,       distance,  0.0),
        ( 0.0,      -distance,  0.0),
        ( 0.0,       0.0,       distance),
        ( 0.0,       0.0,      -distance),
    ]
def octahedral_positions_bidentate(sites):
    return [(sites[0],sites[2]),(sites[1],sites[4]),(sites[3],sites[5])]


def octahedral_positions_tridentate(sites):
    #return [(sites[0],sites[2],sites[1]),(sites[3],sites[5],sites[4])]
    return [(sites[0],sites[2],sites[1]),(sites[3],sites[4],sites[5])]

def tetrahedral_positions(distance: float = 2.0) -> list[Position]:
    """Return four positions pointing to the corners of a tetrahedron."""
    a = distance / (3 ** 0.5)
    return [
        ( a,  a,  a),
        ( a, -a, -a),
        (-a,  a, -a),
        (-a, -a,  a),
    ]


def square_planar_positions(distance: float = 2.0) -> list[Position]:
    """Return four positions in the xy plane."""
    return [
        ( distance, 0.0, 0.0),
        (-distance, 0.0, 0.0),
        ( 0.0,  distance, 0.0),
        ( 0.0, -distance, 0.0),
    ]


def square_antiprismatic_positions(distance: float = 2.0) -> list[Position]:
    """Return eight sites arranged as two staggered squares."""
    z = distance * 0.45
    radius = (distance * distance - z * z) ** 0.5
    positions = []
    for z_value, offset in ((z, math.pi / 4), (-z, 0.0)):
        for i in range(4):
            angle = offset + i * math.pi / 2
            positions.append(
                (
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    z_value,
                )
            )
    return positions


def linear_positions(distance: float = 2.0) -> list[Position]:
    """Return two collinear positions along x."""
    return [( distance, 0.0, 0.0), (-distance, 0.0, 0.0)]


def trigonal_planar_positions(distance: float = 2.0) -> list[Position]:
    """Return three positions in the xy plane, 120° apart."""
    import math
    return [
        (distance * math.cos(math.radians(angle)),
         distance * math.sin(math.radians(angle)),
         0.0)
        for angle in (0, 120, 240)
    ]


def trigonal_bipyramidal_positions(distance: float = 2.0) -> list[Position]:
    """Three equatorial + two axial positions."""
    return trigonal_planar_positions(distance) + [
        (0.0, 0.0,  distance),
        (0.0, 0.0, -distance),
    ]

def square_antiprismatic_positions(distance: float = 2.0) -> list[Position]:
    """Return eight positions for a square antiprismatic geometry."""
    import math

    z = distance * 0.55
    r = (distance**2 - z**2) ** 0.5

    bottom_square = []
    top_square = []

    for i in range(4):
        angle = math.pi / 2 * i

        bottom_square.append(
            (
                r * math.cos(angle),
                r * math.sin(angle),
                -z,
            )
        )

    for i in range(4):
        angle = math.pi / 4 + math.pi / 2 * i

        top_square.append(
            (
                r * math.cos(angle),
                r * math.sin(angle),
                z,
            )
        )

    return bottom_square + top_square


def dodecahedral_positions(distance: float = 2.0) -> list[Position]:
    """Return eight approximate positions for a dodecahedral CN=8 geometry.

    This represents the coordination-chemistry triangular dodecahedron
    (D2d-like), not a regular carbon-style dodecahedron.  The sites are
    arranged in two unequal z levels on each side of the metal so it stays
    visually distinct from the two parallel squares of a square antiprism.
    """
    import math

    z_high = distance * 0.72
    z_low = distance * 0.30

    r_high = math.sqrt(distance**2 - z_high**2)
    r_low = math.sqrt(distance**2 - z_low**2)

    return [
        ( r_high,  0.0,     z_high),
        ( 0.0,     r_low,   z_low),
        (-r_high,  0.0,     z_high),
        ( 0.0,    -r_low,   z_low),
        ( 0.0,    -r_high, -z_high),
        ( r_low,   0.0,    -z_low),
        ( 0.0,     r_high, -z_high),
        (-r_low,   0.0,    -z_low),
    ]
def pentagonal_bipyramidal_positions(distance: float = 2.0) -> list[Position]:
    """Return seven positions: five equatorial + two axial."""
    import math

    equatorial = [
        (
            distance * math.cos(2 * math.pi * i / 5),
            distance * math.sin(2 * math.pi * i / 5),
            0.0,
        )
        for i in range(5)
    ]

    axial = [
        (0.0, 0.0, distance),
        (0.0, 0.0, -distance),
    ]

    return equatorial + axial

def capped_octahedral_positions(distance: float = 2.0) -> list[Position]:
    """Return seven positions for an approximate capped octahedral geometry."""
    base = octahedral_positions(distance)

    cap = (
        distance / (3 ** 0.5),
        distance / (3 ** 0.5),
        distance / (3 ** 0.5),
    )

    return base + [cap]

def square_pyramidal_positions(distance: float = 2.0) -> list[Position]:
    """Return five positions for a square pyramidal geometry."""
    return [
        ( distance,  0.0, 0.0),
        (-distance,  0.0, 0.0),
        ( 0.0,  distance, 0.0),
        ( 0.0, -distance, 0.0),
        ( 0.0,  0.0, distance),
    ]

# Mapping from geometry label (as produced by ``coordchem.geometry``) to
# the corresponding position generator. ``geometry_positions`` does a
# best-effort lookup so that ambiguous labels like
# "trigonal bipyramidal or square pyramidal" still produce something
# reasonable.
_GEOMETRY_BUILDERS = {
    "linear": linear_positions,
    "trigonal planar": trigonal_planar_positions,
    "tetrahedral": tetrahedral_positions,
    "square planar": square_planar_positions,
    "square antiprismatic": square_antiprismatic_positions,
    "trigonal bipyramidal": trigonal_bipyramidal_positions,
    "square pyramidal": square_pyramidal_positions,
    "pentagonal bipyramidal": pentagonal_bipyramidal_positions,
    "capped octahedral": capped_octahedral_positions,   
    "octahedral": octahedral_positions,
    "square antiprismatic": square_antiprismatic_positions,
    "antiprismatic": square_antiprismatic_positions,
    "dodecahedral":dodecahedral_positions,
   
}


def geometry_positions(
    geometry: str | None,
    n: int,
    distance: float = 2.0,
) -> list[Position]:
    """
    Return ``n`` coordination-site positions matching ``geometry``.

    Falls back to an octahedral arrangement (truncated/extended to ``n``)
    when the geometry label is unknown — useful as a TODO placeholder for
    geometries we have not implemented yet.
    """
    if geometry:
        for label, builder in _GEOMETRY_BUILDERS.items():
            if label in geometry.lower():
                positions = builder(distance)
                if len(positions) >= n:
                    return positions[:n]
                # not enough sites: extend with octahedral ones
                extra = octahedral_positions(distance)[: n - len(positions)]
                return positions + extra

    # if the geometry is unknown the octahedral one is adopted by default. 
    # For n ligands, if n is < to the number of ligand in the octahedral complex only the n first positions are attributed. 
    base = octahedral_positions(distance)
    if n <= len(base):
        return base[:n]
    return base + [(0.0, 0.0, 0.0)] * (n - len(base))
    #if n > o the number of ligand in the octahedral complex the 2 extra ligands fall in the center (0.0, 0.0, 0.0)


def bidentate_site_pair_indices(
    geometry: str | None,
    n_sites: int,
    *,
    short_bidentate: bool = True,
) -> list[tuple[int, int]]:
    """Return chemically reasonable site pairs for bidentate ligands."""
    if not short_bidentate:
        return [(i, i + 1) for i in range(0, n_sites - 1, 2)]

    normalized = (geometry or "").lower()

    if "pentagonal bipyramidal" in normalized and n_sites >= 7:
        return [
            (0, 1),
            (2, 3),
            (3, 4),
            (4, 0),
            (1, 2),
            (4, 5),
            (5, 2),
            (6, 1),
            (6, 3),
        ]

    if "capped octahedral" in normalized and n_sites >= 7:
        return [(0, 2), (1, 4), (3, 5), (0, 6), (2, 6), (4, 6)]

    if "octahedral" in normalized and n_sites >= 6:
        return [(0, 2), (1, 4), (3, 5)]

    if "square planar" in normalized and n_sites >= 4:
        return [(0, 2), (1, 3)]

    if "square antiprismatic" in normalized and n_sites >= 8:
        return [(0, 1), (2, 3), (4, 5), (6, 7)]

    if "dodecahedral" in normalized and n_sites >= 8:
        return [(0, 1), (2, 3), (4, 5), (6, 7)]

    return [(i, i + 1) for i in range(0, n_sites - 1, 2)]


# ---------------------------------------------------------------------------
# Ligand-level helpers
# ---------------------------------------------------------------------------

def build_ligand_3d(smiles: str) -> Chem.Mol:
    """
    Build an RDKit ``Mol`` from ``smiles`` with explicit Hs and a
    single 3D conformer.

    Raises
    ------
    ValueError
        If the SMILES cannot be parsed or a 3D conformer cannot be embedded.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")

    mol = Chem.AddHs(mol)

    # Ions like [Cl-] or [O-2] have a single atom — nothing to embed.
    if mol.GetNumAtoms() <= 1:
        conf = Chem.Conformer(mol.GetNumAtoms())
        if mol.GetNumAtoms() == 1:
            conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
        mol.AddConformer(conf, assignId=True)
        return mol

    status = AllChem.EmbedMolecule(mol, randomSeed=42)
    if status != 0:
        # Retry with random coordinates as fallback
        status = AllChem.EmbedMolecule(
            mol, randomSeed=42, useRandomCoords=True
        )
        if status != 0:
            raise ValueError(f"3D embedding failed for SMILES: {smiles!r}")

    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        # Optimization is best-effort: keep the embedded geometry
        pass

    return mol


def find_donor_atom(mol, donor_symbol, override: Optional[int] = None):
    if override is not None and 0 <= override < mol.GetNumAtoms():
        return override

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == donor_symbol:
            return atom.GetIdx()

    raise ValueError(f"No donor atom {donor_symbol} found in ligand")

        
 #function for the ligands disposition around the metal


def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def vec_scale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)

def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )

def norm(v):
    return (v[0]**2 + v[1]**2 + v[2]**2) ** 0.5

def unit(v):
    n = norm(v)
    if n == 0:
        return (1.0, 0.0, 0.0)
    return (v[0]/n, v[1]/n, v[2]/n)

def rotate_vector(v, axis, angle):
    import math

    axis = unit(axis)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    term1 = vec_scale(v, cos_a)
    term2 = vec_scale(cross(axis, v), sin_a)
    term3 = vec_scale(axis, dot(axis, v) * (1 - cos_a))

    return vec_add(vec_add(term1, term2), term3)

def translate_bidentate_ligand(ligand_mol, donor_indices, target_sites):
    ligand_conf = ligand_mol.GetConformer()

    donor1 = ligand_conf.GetAtomPosition(donor_indices[0])
    donor2 = ligand_conf.GetAtomPosition(donor_indices[1])

    donor1 = (donor1.x, donor1.y, donor1.z)
    donor2 = (donor2.x, donor2.y, donor2.z)

    ligand_mid = (
    (donor1[0] + donor2[0]) / 2,
    (donor1[1] + donor2[1]) / 2,
    (donor1[2] + donor2[2]) / 2,
)

    target1, target2 = target_sites

    target_mid = (
    (target1[0] + target2[0]) / 2,
    (target1[1] + target2[1]) / 2,
    (target1[2] + target2[2]) / 2,
)
    push= 2.0
    outward=unit(target_mid)
    
    target_mid = (
        target_mid[0] + outward[0] * push,
        target_mid[1] + outward[1] * push,
        target_mid[2] + outward[2] * push,
    )

    ligand_axis = unit(vec_sub(donor2, donor1))
    target_axis = unit(vec_sub(target2, target1))

    rotation_axis=cross(ligand_axis, target_axis)
    rotation_axis_norm=norm(rotation_axis)

    if rotation_axis_norm == 0:
        angle = 0.0
        rotation_axis = (1.0, 0.0, 0.0)
    else:
        cos_angle = max(-1.0, min(1.0, dot(ligand_axis, target_axis)))
        angle = math.acos(cos_angle)

    coords = {}
    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        old_pos = ligand_conf.GetAtomPosition(idx)
        old_pos = (old_pos.x, old_pos.y, old_pos.z)

        centered = vec_sub(old_pos, ligand_mid)
        rotated = rotate_vector(centered, rotation_axis, angle)
        final_pos = vec_add(rotated, target_mid)

        coords[idx] = final_pos
    return coords
       
def translate_tridentate_ligand(ligand_mol, donor_indices, target_sites, reverse):
    ligand_conf = ligand_mol.GetConformer()

    donor_positions=[]
    for idx in donor_indices:
        pos=ligand_conf.GetAtomPosition(idx)
        donor_positions.append((pos.x,pos.y,pos.z))

    ligand_mid=(sum(pos[0] for pos in donor_positions)/3,
                sum(pos[1] for pos in donor_positions)/3,
                sum(pos[2] for pos in donor_positions)/3
                    )
    target_mid=(sum(pos[0] for pos in target_sites)/3,
                sum(pos[1] for pos in target_sites)/3,
                sum(pos[2] for pos in target_sites)/3
                    )
    outward=unit(target_mid)
    push=0.8

    target_mid=(target_mid[0] + outward[0] * push,
        target_mid[1] + outward[1] * push,
        target_mid[2] + outward[2] * push)
    coords={}
    for atom in ligand_mol.GetAtoms():
        idx=atom.GetIdx()
        old_pos=ligand_conf.GetAtomPosition(idx)
        old_pos=(old_pos.x, old_pos.y, old_pos.z)
        centered=vec_sub(old_pos, ligand_mid)
        if reverse:
            centered=(-centered[0], -centered[1], -centered[2])

        final_pos=vec_add(centered, target_mid)
        coords[idx]=final_pos

    return coords

def translate_tetradentate_ligand(ligand_mol, donor_indices, target_sites):
    ligand_conf = ligand_mol.GetConformer()

    donor_positions=[]
    for idx in donor_indices:
        pos=ligand_conf.GetAtomPosition(idx)
        donor_positions.append((pos.x,pos.y,pos.z))

    ligand_mid=(sum(pos[0] for pos in donor_positions)/4,
                sum(pos[1] for pos in donor_positions)/4,
                sum(pos[2] for pos in donor_positions)/4
                    )
    target_mid=(sum(pos[0] for pos in target_sites)/4,
                sum(pos[1] for pos in target_sites)/4,
                sum(pos[2] for pos in target_sites)/4
                    )
    
    ligand_x=unit(vec_sub(donor_positions[1], donor_positions[0]))
    ligand_tmp=unit(vec_sub(donor_positions[2], donor_positions[0]))
    ligand_z=unit(cross(ligand_x, ligand_tmp))
    ligand_y=unit(cross(ligand_z, ligand_x))


    target_x = unit(vec_sub(target_sites[1], target_sites[0]))
    target_tmp = unit(vec_sub(target_sites[2], target_sites[0]))
    target_z = unit(cross(target_x, target_tmp))
    target_y = unit(cross(target_z, target_x))

    coords={}
    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        old_pos = ligand_conf.GetAtomPosition(idx)
        old_pos = (old_pos.x, old_pos.y, old_pos.z)

        centered = vec_sub(old_pos, ligand_mid)

        x = dot(centered, ligand_x)
        y = dot(centered, ligand_y)
        z = dot(centered, ligand_z)

        final_pos = vec_add(
            target_mid,
            vec_add(
                vec_add(vec_scale(target_x, x), vec_scale(target_y, y)),
                vec_scale(target_z, z),
            ),
        )

        coords[idx] = final_pos

    return coords

def translate_hexadentate_ligand(ligand_mol, donor_indices, target_sites):
    ligand_conf = ligand_mol.GetConformer()

    donor_positions = []
    for idx in donor_indices:
        pos = ligand_conf.GetAtomPosition(idx)
        donor_positions.append((pos.x, pos.y, pos.z))

    ligand_mid = (
        sum(pos[0] for pos in donor_positions) / 6,
        sum(pos[1] for pos in donor_positions) / 6,
        sum(pos[2] for pos in donor_positions) / 6,
    )

    target_mid = (
        sum(pos[0] for pos in target_sites) / 6,
        sum(pos[1] for pos in target_sites) / 6,
        sum(pos[2] for pos in target_sites) / 6,
    )
    donor_shift={}
    for donor_idx, old_donor_pos, target_pos in zip(donor_indices, donor_positions,target_sites):
        donor_shift[donor_idx]=vec_sub(target_pos,old_donor_pos)
    coords = {}

    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        old_pos = ligand_conf.GetAtomPosition(idx)
        old_pos = (old_pos.x, old_pos.y, old_pos.z)
        if idx in donor_shift:
            coords[idx]=vec_add(old_pos,donor_shift[idx])
        else:
            centered = vec_sub(old_pos, ligand_mid)
            final_pos = vec_add(centered, target_mid)
            coords[idx] = final_pos

    return coords


# ---------------------------------------------------------------------------
# Complex-level builder
# ---------------------------------------------------------------------------

def thiocyanato_ligand_positions(ligand_mol, donor_idx,target,ligand_symbol):
    direction = unit(target)

    center = target
    coords = {}
    coords[donor_idx] = center

    if abs(direction[0]) < 0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)

    n_indices = [
        atom.GetIdx()
        for atom in ligand_mol.GetAtoms()
        if atom.GetSymbol() == "N"]
    s_indices = [
        atom.GetIdx()
        for atom in ligand_mol.GetAtoms()
        if atom.GetSymbol() == "S"] 
    c_indices = [
        atom.GetIdx()
        for atom in ligand_mol.GetAtoms()
        if atom.GetSymbol() == "C"] 

    cn_length = 1.16
    cs_length = 1.63

    if ligand_symbol=="NCS":
        n_idx=donor_idx
        coords[n_idx]=target

        c_idx=c_indices[0]
        c_pos=vec_add(target,vec_scale(direction,cn_length))
        coords[c_idx]=c_pos
            
        s_idx=s_indices[0]
        s_pos=vec_add(c_pos,vec_scale(direction,cs_length))
        coords[s_idx]=s_pos

    elif ligand_symbol=="SCN":
        s_idx=donor_idx
        coords[s_idx]=target

        c_idx=c_indices[0]
        c_pos=vec_add(target,vec_scale(direction,cs_length))
        coords[c_idx]=c_pos
            
        n_idx=n_indices[0]
        n_pos=vec_add(c_pos,vec_scale(direction,cn_length))
        coords[n_idx]=n_pos

    for atom in ligand_mol.GetAtoms():
        idx= atom.GetIdx()
        if idx not in coords:
            coords[idx]=target
    return coords


def azido_ligand_positions(ligand_mol, donor_idx, target):
    """Place azide as a linear M-N-N-N ligand pointing away from the metal."""
    direction = unit(target)
    coords = {donor_idx: target}

    nn_length = 1.18
    current_idx = donor_idx
    current_pos = target
    seen = {donor_idx}

    while True:
        next_n = next(
            (
                neighbor.GetIdx()
                for neighbor in ligand_mol.GetAtomWithIdx(current_idx).GetNeighbors()
                if neighbor.GetSymbol() == "N" and neighbor.GetIdx() not in seen
            ),
            None,
        )
        if next_n is None:
            break

        current_pos = vec_add(current_pos, vec_scale(direction, nn_length))
        coords[next_n] = current_pos
        seen.add(next_n)
        current_idx = next_n

    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in coords:
            coords[idx] = target

    return coords

    
def nitrito_ligand_positions(ligand_mol, donor_idx, target, ligand_symbol):
    direction = unit(target)

    coords = {donor_idx: target}

    if abs(direction[0]) < 0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)

    u = unit(cross(direction, ref))
    no_single_length = 1.30
    no_double_length = 1.22
    no_resonance_length = 1.24

    n_indices = [
        atom.GetIdx()
        for atom in ligand_mol.GetAtoms()
        if atom.GetSymbol() == "N"]
    o_indices = [
        atom.GetIdx()
        for atom in ligand_mol.GetAtoms()
        if atom.GetSymbol() == "O"]
    if ligand_symbol=="NO2":
        n_idx=donor_idx
        coords[n_idx]=target

        if len(o_indices)>=2:
            side_component = (3.0 ** 0.5) / 2.0
            o_dir_1 = vec_add(
                vec_scale(direction, 0.5),
                vec_scale(u, side_component),
            )
            o_dir_2 = vec_add(
                vec_scale(direction, 0.5),
                vec_scale(u, -side_component),
            )
            coords[o_indices[0]]=vec_add(target,vec_scale(o_dir_1, no_resonance_length))
            coords[o_indices[1]]=vec_add(target,vec_scale(o_dir_2, no_resonance_length))

    elif ligand_symbol=="ONO":
        o_idx=donor_idx
        coords[o_idx]=target

        n_direction = unit(
            vec_add(
                vec_scale(direction, 0.5),
                vec_scale(u, (3.0 ** 0.5) / 2.0),
            )
        )
        coords[n_indices[0]]=vec_add(target,vec_scale(n_direction, no_single_length))
        other_o=[idx for idx in o_indices if idx!=donor_idx]
        if other_o:
            coords[other_o[0]]=vec_add(
                coords[n_indices[0]],
                vec_scale(direction, no_double_length),
            )
    for atom in ligand_mol.GetAtoms():
        idx= atom.GetIdx()
        if idx not in coords:
            coords[idx]=target
    return coords

def methyl_ligand_positions(ligand_mol, donor_idx, target, nh_distance=1.0, spread=0.8):
    direction = unit(target)

    center = target
    coords = {}
    coords[donor_idx] = center

    if abs(direction[0]) < 0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)

    u = unit(cross(direction, ref))
    v = unit(cross(direction, u))

    h_indices = [
        atom.GetIdx()
        for atom in ligand_mol.GetAtoms()
        if atom.GetSymbol() == "H"
    ]

    ch_length = 1.09
    for h_idx, h_direction in zip(
        h_indices[:3],
        _tetrahedral_substituent_directions(direction, u, v),
    ):
        coords[h_idx] = vec_add(center, vec_scale(h_direction, ch_length))

    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in coords:
            coords[idx] = center

    return coords
def ammonia_ligand_positions(ligand_mol, donor_idx, target, nh_distance=1.0, spread=0.8):
    direction = unit(target)

    center = target
    coords = {}
    coords[donor_idx] = center

    if abs(direction[0]) < 0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)

    u = unit(cross(direction, ref))
    v = unit(cross(direction, u))

    h_indices = [
        atom.GetIdx()
        for atom in ligand_mol.GetAtoms()
        if atom.GetSymbol() == "H"
    ]

    nh_length = 1.01
    for h_idx, h_direction in zip(
        h_indices[:3],
        _tetrahedral_substituent_directions(direction, u, v),
    ):
        coords[h_idx] = vec_add(center, vec_scale(h_direction, nh_length))

    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in coords:
            coords[idx] = center

    return coords


def pyridine_ligand_positions(ligand_mol, donor_idx_local,target,ligand_number=0):
    direction = unit(target)

    metal_n_length = max(norm(target), 3.0)
    center = vec_scale(direction, metal_n_length)
    coords = {}
    coords[donor_idx_local] = center

    if abs(direction[0]) < 0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)

    u = unit(cross(direction, ref))
    v = unit(cross(direction, u))
    aromatic_bond_length = 1.39
    ch_length = 1.09
    ring = next(
        (
            ring
            for ring in ligand_mol.GetRingInfo().AtomRings()
            if donor_idx_local in ring and len(ring) == 6
        ),
        (),
    )
    heavy_atoms = _ordered_ring_from_anchor(ligand_mol, ring, donor_idx_local)
    ring_center=vec_add(center,vec_scale(direction, aromatic_bond_length))
    coords[donor_idx_local]=center
    angles=[math.pi, 2*math.pi/3, math.pi/3,0.0,-math.pi/3,-2*math.pi/3]

    for idx, angle in zip(heavy_atoms, angles):
        ring_pos = vec_add(
                vec_scale(direction, math.cos(angle) * aromatic_bond_length),
                vec_scale(u, math.sin(angle) * aromatic_bond_length))
        coords[idx] = vec_add(ring_center, ring_pos)
    for atom in ligand_mol.GetAtoms():
        idx=atom.GetIdx()
        if idx in coords:
            continue 
        if atom.GetSymbol() == "H":
            neighbors = atom.GetNeighbors()
            if neighbors:
                heavy_idx = neighbors[0].GetIdx()
                base = coords.get(heavy_idx, center)
                outward=unit(vec_sub(base, ring_center))
                coords[idx]=vec_add(base, vec_scale(outward,ch_length))
            else:
                coords[idx]=center
    return coords



def _tetrahedral_substituent_directions(direction, u, v):
    tetrahedral_radial = (8.0 / 9.0) ** 0.5
    directions = []
    for i in range(3):
        angle = 2 * math.pi * i / 3
        radial = vec_add(
            vec_scale(u, math.cos(angle)),
            vec_scale(v, math.sin(angle)),
        )
        directions.append(
            unit(
                vec_add(
                    vec_scale(direction, 1.0 / 3.0),
                    vec_scale(radial, tetrahedral_radial),
                )
            )
        )
    return directions


def _perpendicular_unit(reference, preferred):
    side = vec_sub(preferred, vec_scale(reference, dot(preferred, reference)))
    if norm(side) > 1e-8:
        return unit(side)
    side = cross(reference, (1.0, 0.0, 0.0))
    if norm(side) > 1e-8:
        return unit(side)
    return unit(cross(reference, (0.0, 1.0, 0.0)))


def _place_methyl_hydrogens(coords, ligand_mol, carbon_idx, heavy_idx, side_reference):
    carbon_pos = coords[carbon_idx]
    heavy_direction = unit(vec_sub(coords[heavy_idx], carbon_pos))
    side = _perpendicular_unit(heavy_direction, side_reference)
    lift = unit(cross(heavy_direction, side))
    ch_length = 1.09

    h_indices = [
        neighbor.GetIdx()
        for neighbor in ligand_mol.GetAtomWithIdx(carbon_idx).GetNeighbors()
        if neighbor.GetSymbol() == "H"
    ]
    for i, h_idx in enumerate(h_indices):
        angle = 2 * math.pi * i / max(1, len(h_indices))
        h_direction = vec_add(
            vec_scale(heavy_direction, -1.0 / 3.0),
            vec_scale(
                vec_add(
                    vec_scale(side, math.cos(angle)),
                    vec_scale(lift, math.sin(angle)),
                ),
                (8.0 / 9.0) ** 0.5,
            ),
        )
        coords[h_idx] = vec_add(carbon_pos, vec_scale(unit(h_direction), ch_length))


def _place_methylene_hydrogens(coords, ligand_mol, carbon_idx, heavy_indices):
    carbon_pos = coords[carbon_idx]
    heavy_dirs = [unit(vec_sub(coords[idx], carbon_pos)) for idx in heavy_indices]
    if len(heavy_dirs) < 2:
        return

    bisector = unit(vec_scale(vec_add(heavy_dirs[0], heavy_dirs[1]), -1.0))
    normal = cross(heavy_dirs[0], heavy_dirs[1])
    if norm(normal) <= 1e-8:
        normal = _perpendicular_unit(bisector, (0.0, 0.0, 1.0))
    normal = unit(normal)
    ch_length = 1.09

    h_indices = [
        neighbor.GetIdx()
        for neighbor in ligand_mol.GetAtomWithIdx(carbon_idx).GetNeighbors()
        if neighbor.GetSymbol() == "H"
    ]
    h_dirs = (
        vec_add(
            vec_scale(bisector, 1.0 / (3.0 ** 0.5)),
            vec_scale(normal, (2.0 / 3.0) ** 0.5),
        ),
        vec_add(
            vec_scale(bisector, 1.0 / (3.0 ** 0.5)),
            vec_scale(normal, -(2.0 / 3.0) ** 0.5),
        ),
    )
    for h_idx, h_direction in zip(h_indices, h_dirs):
        coords[h_idx] = vec_add(carbon_pos, vec_scale(unit(h_direction), ch_length))


def _place_two_tetrahedral_hydrogens(
    coords,
    ligand_mol,
    parent_idx,
    heavy_positions,
    bond_length,
):
    """Place two H atoms around a parent with two fixed heavy substituents."""
    parent_pos = coords[parent_idx]
    heavy_dirs = [unit(vec_sub(position, parent_pos)) for position in heavy_positions]
    if len(heavy_dirs) < 2:
        return

    h_indices = [
        neighbor.GetIdx()
        for neighbor in ligand_mol.GetAtomWithIdx(parent_idx).GetNeighbors()
        if neighbor.GetSymbol() == "H"
    ]
    if len(h_indices) < 2:
        return

    heavy_sum = vec_add(heavy_dirs[0], heavy_dirs[1])
    if norm(heavy_sum) <= 1e-8:
        bisector = _perpendicular_unit(heavy_dirs[0], (0.0, 0.0, 1.0))
        bisector_component = 0.0
    else:
        bisector = unit(vec_scale(heavy_sum, -1.0))
        bisector_component = max(-1.0, min(1.0, norm(heavy_sum) / 2.0))

    normal = cross(heavy_dirs[0], heavy_dirs[1])
    if norm(normal) <= 1e-8:
        normal = _perpendicular_unit(bisector, (0.0, 0.0, 1.0))
    normal = unit(normal)
    normal_component = max(0.0, 1.0 - bisector_component * bisector_component) ** 0.5

    h_dirs = (
        vec_add(
            vec_scale(bisector, bisector_component),
            vec_scale(normal, normal_component),
        ),
        vec_add(
            vec_scale(bisector, bisector_component),
            vec_scale(normal, -normal_component),
        ),
    )

    for h_idx, h_direction in zip(h_indices[:2], h_dirs):
        coords[h_idx] = vec_add(parent_pos, vec_scale(unit(h_direction), bond_length))


def trimethyl_ligand_positions(ligand_mol, donor_idx, target, nh_distance=1.0, spread=0.8):
    direction = unit(target)

    metal_p_length = max(norm(target), 4.0)
    center = vec_scale(direction, metal_p_length)
    coords = {}
    coords[donor_idx] = center

    if abs(direction[0]) < 0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)

    u = unit(cross(direction, ref))
    v = unit(cross(direction, u))

    p_atom = ligand_mol.GetAtomWithIdx(donor_idx)
    methyl_indices = [
        neighbor.GetIdx()
        for neighbor in p_atom.GetNeighbors()
        if neighbor.GetSymbol() == "C"
    ]

    pc_length = 1.83
    substituent_directions = _tetrahedral_substituent_directions(direction, u, v)
    for c_idx, pc_direction in zip(methyl_indices[:3], substituent_directions):
        coords[c_idx] = vec_add(center, vec_scale(pc_direction, pc_length))
        _place_methyl_hydrogens(coords, ligand_mol, c_idx, donor_idx, direction)

    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in coords:
            coords[idx] = center

    return coords


def triphenylphosphine_ligand_positions(ligand_mol, donor_idx, target):
    """Place PPh3 with tetrahedral phosphorus and outward phenyl rings."""
    direction = unit(target)
    metal_p_length = max(norm(target), 4.0)
    center = vec_scale(direction, metal_p_length)
    coords = {donor_idx: center}

    if abs(direction[0]) < 0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)

    u = unit(cross(direction, ref))
    v = unit(cross(direction, u))

    p_atom = ligand_mol.GetAtomWithIdx(donor_idx)
    ipso_carbons = [
        neighbor.GetIdx()
        for neighbor in p_atom.GetNeighbors()
        if neighbor.GetSymbol() == "C"
    ]

    pc_length = 1.83
    aromatic_cc_length = 1.39
    ch_length = 1.09
    tetrahedral_radial = (8.0 / 9.0) ** 0.5

    ring_info = ligand_mol.GetRingInfo()
    rings_by_ipso = {}
    for ring in ring_info.AtomRings():
        ring_set = set(ring)
        for ipso_idx in ipso_carbons:
            if ipso_idx in ring_set:
                rings_by_ipso[ipso_idx] = ring
                break

    for i, ipso_idx in enumerate(ipso_carbons[:3]):
        angle = 2 * math.pi * i / 3
        radial = vec_add(
            vec_scale(u, math.cos(angle)),
            vec_scale(v, math.sin(angle)),
        )
        pc_direction = unit(
            vec_add(
                vec_scale(direction, 1.0 / 3.0),
                vec_scale(radial, tetrahedral_radial),
            )
        )

        ipso_pos = vec_add(center, vec_scale(pc_direction, pc_length))
        coords[ipso_idx] = ipso_pos

        ring = rings_by_ipso.get(ipso_idx)
        if ring is None:
            continue

        ring_side = cross(pc_direction, direction)
        if norm(ring_side) <= 1e-8:
            ring_side = cross(pc_direction, u)
        ring_side = unit(ring_side)

        ordered_ring = _ordered_ring_from_anchor(ligand_mol, ring, ipso_idx)
        ring_center = vec_add(ipso_pos, vec_scale(pc_direction, aromatic_cc_length))
        center_to_ipso = vec_scale(pc_direction, -1.0)

        for ring_pos_index, atom_idx in enumerate(ordered_ring):
            theta = ring_pos_index * math.pi / 3
            atom_direction = vec_add(
                vec_scale(center_to_ipso, math.cos(theta)),
                vec_scale(ring_side, math.sin(theta)),
            )
            coords[atom_idx] = vec_add(
                ring_center,
                vec_scale(atom_direction, aromatic_cc_length),
            )

        for atom_idx in ordered_ring:
            atom = ligand_mol.GetAtomWithIdx(atom_idx)
            h_neighbors = [
                neighbor.GetIdx()
                for neighbor in atom.GetNeighbors()
                if neighbor.GetSymbol() == "H"
            ]
            if not h_neighbors:
                continue

            atom_pos = coords[atom_idx]
            h_direction = unit(vec_sub(atom_pos, ring_center))
            for h_idx in h_neighbors:
                coords[h_idx] = vec_add(atom_pos, vec_scale(h_direction, ch_length))

    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in coords:
            coords[idx] = center

    return coords


def _ordered_ring_from_anchor(ligand_mol, ring, anchor_idx):
    ring_set = set(ring)
    ring_neighbors = [
        neighbor.GetIdx()
        for neighbor in ligand_mol.GetAtomWithIdx(anchor_idx).GetNeighbors()
        if neighbor.GetIdx() in ring_set
    ]
    if not ring_neighbors:
        return [anchor_idx] + [idx for idx in ring if idx != anchor_idx]

    ordered = [anchor_idx]
    previous_idx = anchor_idx
    current_idx = ring_neighbors[0]
    while len(ordered) < len(ring):
        ordered.append(current_idx)
        next_neighbors = [
            neighbor.GetIdx()
            for neighbor in ligand_mol.GetAtomWithIdx(current_idx).GetNeighbors()
            if neighbor.GetIdx() in ring_set and neighbor.GetIdx() != previous_idx
        ]
        if not next_neighbors:
            break
        previous_idx, current_idx = current_idx, next_neighbors[0]

    if len(ordered) != len(ring):
        ordered = [anchor_idx] + [idx for idx in ring if idx != anchor_idx]
    return ordered


def oxalate_ligand_positions(ligand_mol, donor_indices, target_sites):
    """Place oxalate as a planar O-C(=O)-C(=O)-O chelate."""
    coords={}
    o1_idx, o2_idx = donor_indices[:2]
    o1_pos, o2_pos = target_sites

    coords[o1_idx] = o1_pos
    coords[o2_idx] = o2_pos
    axis=unit(vec_sub(o2_pos, o1_pos))
    mid = (
        (o1_pos[0] + o2_pos[0]) / 2,
        (o1_pos[1] + o2_pos[1]) / 2,
        (o1_pos[2] + o2_pos[2]) / 2,
    )
    outward=unit(mid)

    if norm(outward) == 0:
        outward = (0.0, 0.0, 1.0)
    outward = vec_sub(outward, vec_scale(axis, dot(outward, axis)))
    if norm(outward) == 0:
        outward = (0.0, 0.0, 1.0)
    outward = unit(outward)

    side = cross(axis, outward)

    if norm(side) == 0:
        side = (0.0, 0.0, 1.0)

    side = unit(side)
    o1_atom=ligand_mol.GetAtomWithIdx(o1_idx)
    o2_atom=ligand_mol.GetAtomWithIdx(o2_idx)

    c1_candidates = [
    nbr.GetIdx()
    for nbr in o1_atom.GetNeighbors()
    if nbr.GetSymbol() == "C"
]

    c2_candidates = [
    nbr.GetIdx()
    for nbr in o2_atom.GetNeighbors()
    if nbr.GetSymbol() == "C"
]
    c1_idx = c1_candidates[0]
    c2_idx = c2_candidates[0]

    donor_co_length = 1.27
    carbonyl_length = 1.22
    cc_length = 1.54
    donor_distance = norm(vec_sub(o2_pos, o1_pos))
    axis_shift = max(
        0.0,
        min(donor_co_length * 0.95, (donor_distance - cc_length) / 2.0),
    )
    outward_shift = (donor_co_length**2 - axis_shift**2) ** 0.5

    coords[c1_idx] = vec_add(
        o1_pos,
        vec_add(vec_scale(axis, axis_shift), vec_scale(outward, outward_shift)),
    )
    coords[c2_idx] = vec_add(
        o2_pos,
        vec_add(vec_scale(axis, -axis_shift), vec_scale(outward, outward_shift)),
    )

    carbonyl_1 = [
        nbr.GetIdx()
        for nbr in ligand_mol.GetAtomWithIdx(c1_idx).GetNeighbors()
        if nbr.GetSymbol() == "O" and nbr.GetIdx() != o1_idx
    ]
    carbonyl_2 = [
        nbr.GetIdx()
        for nbr in ligand_mol.GetAtomWithIdx(c2_idx).GetNeighbors()
        if nbr.GetSymbol() == "O" and nbr.GetIdx() != o2_idx
    ]
    if carbonyl_1:
        donor_direction = unit(vec_sub(o1_pos, coords[c1_idx]))
        cc_direction = unit(vec_sub(coords[c2_idx], coords[c1_idx]))
        carbonyl_direction = unit(vec_scale(vec_add(donor_direction, cc_direction), -1.0))
        coords[carbonyl_1[0]] = vec_add(
            coords[c1_idx],
            vec_scale(carbonyl_direction, carbonyl_length),
        )
    if carbonyl_2:
        donor_direction = unit(vec_sub(o2_pos, coords[c2_idx]))
        cc_direction = unit(vec_sub(coords[c1_idx], coords[c2_idx]))
        carbonyl_direction = unit(vec_scale(vec_add(donor_direction, cc_direction), -1.0))
        coords[carbonyl_2[0]] = vec_add(
            coords[c2_idx],
            vec_scale(carbonyl_direction, carbonyl_length),
        )

    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()

        if idx in coords:
            continue

        coords[idx] = mid

    return coords


def acac_ligand_positions(ligand_mol, donor_indices, target_sites):
    """Place acetylacetonate with sp2 planar and sp3 tetrahedral geometry."""
    coords = {}
    o1_idx, o2_idx = donor_indices[:2]
    o1_pos, o2_pos = target_sites

    coords[o1_idx] = o1_pos
    coords[o2_idx] = o2_pos

    o1_atom = ligand_mol.GetAtomWithIdx(o1_idx)
    o2_atom = ligand_mol.GetAtomWithIdx(o2_idx)
    c1_idx = next(n.GetIdx() for n in o1_atom.GetNeighbors() if n.GetSymbol() == "C")
    c2_idx = next(n.GetIdx() for n in o2_atom.GetNeighbors() if n.GetSymbol() == "C")

    def perpendicular_unit(reference, preferred):
        side = vec_sub(preferred, vec_scale(reference, dot(preferred, reference)))
        if norm(side) > 1e-8:
            return unit(side)
        side = cross(reference, (1.0, 0.0, 0.0))
        if norm(side) > 1e-8:
            return unit(side)
        return unit(cross(reference, (0.0, 1.0, 0.0)))

    ligand_mid = (
        (o1_pos[0] + o2_pos[0]) / 2,
        (o1_pos[1] + o2_pos[1]) / 2,
        (o1_pos[2] + o2_pos[2]) / 2,
    )
    outward = unit(ligand_mid)
    if norm(outward) <= 1e-8:
        outward = unit(vec_add(o1_pos, o2_pos))
    axis = unit(vec_sub(o2_pos, o1_pos))
    o_o_distance = norm(vec_sub(o2_pos, o1_pos))
    half_o_o = o_o_distance / 2.0

    oc_length = 1.28
    cc_sp2_length = 1.40
    c_methyl_length = 1.50
    ch_length = 1.09

    # Planar beta-diketonate backbone: O-C-C-C-O.
    # The central sp2 carbon sits at the apex of a trigonal-planar C-C-C angle.
    bridge_half_angle_drop = cc_sp2_length / 2.0
    donor_c_axis = (3.0 ** 0.5 / 2.0) * cc_sp2_length
    donor_c_axis = min(donor_c_axis, max(0.05, half_o_o - 0.05))
    oc_axis_component = half_o_o - donor_c_axis
    if abs(oc_axis_component) > oc_length:
        oc_axis_component = max(-oc_length * 0.95, min(oc_length * 0.95, oc_axis_component))
    donor_c_out = (oc_length**2 - oc_axis_component**2) ** 0.5

    c1_pos = vec_add(
        o1_pos,
        vec_add(vec_scale(axis, oc_axis_component), vec_scale(outward, donor_c_out)),
    )
    c2_pos = vec_add(
        o2_pos,
        vec_add(vec_scale(axis, -oc_axis_component), vec_scale(outward, donor_c_out)),
    )
    coords[c1_idx] = c1_pos
    coords[c2_idx] = c2_pos

    bridge_candidates = [
        atom.GetIdx()
        for atom in ligand_mol.GetAtoms()
        if atom.GetSymbol() == "C"
        and atom.GetIdx() not in {c1_idx, c2_idx}
        and any(n.GetIdx() in {c1_idx, c2_idx} for n in atom.GetNeighbors())
        and sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "H") <= 1
    ]
    bridge_idx = bridge_candidates[0] if bridge_candidates else None
    if bridge_idx is not None:
        coords[bridge_idx] = vec_add(
            ligand_mid,
            vec_scale(outward, donor_c_out + bridge_half_angle_drop),
        )

    methyl_indices = []
    for c_idx, donor_c_pos in ((c1_idx, c1_pos), (c2_idx, c2_pos)):
        attached_carbons = [
            n.GetIdx()
            for n in ligand_mol.GetAtomWithIdx(c_idx).GetNeighbors()
            if n.GetSymbol() == "C" and n.GetIdx() != bridge_idx
        ]
        for methyl_idx in attached_carbons:
            methyl_indices.append(methyl_idx)
            to_oxygen = unit(vec_sub(coords[o1_idx if c_idx == c1_idx else o2_idx], donor_c_pos))
            to_bridge = unit(vec_sub(coords.get(bridge_idx, ligand_mid), donor_c_pos))
            methyl_direction = unit(vec_scale(vec_add(to_oxygen, to_bridge), -1.0))
            coords[methyl_idx] = vec_add(
                donor_c_pos,
                vec_scale(methyl_direction, c_methyl_length),
            )

    normal = cross(axis, outward)
    if norm(normal) <= 1e-8:
        normal = perpendicular_unit(axis, (0.0, 0.0, 1.0))
    normal = unit(normal)

    hydrogens_by_parent: dict[int, list[int]] = {}
    for atom in ligand_mol.GetAtoms():
        if atom.GetSymbol() != "H":
            continue
        parent = atom.GetNeighbors()[0].GetIdx()
        hydrogens_by_parent.setdefault(parent, []).append(atom.GetIdx())

    for parent_idx, h_indices in hydrogens_by_parent.items():
        parent_pos = coords.get(parent_idx, ligand_mid)
        heavy_neighbors = [
            n.GetIdx()
            for n in ligand_mol.GetAtomWithIdx(parent_idx).GetNeighbors()
            if n.GetSymbol() != "H" and n.GetIdx() in coords
        ]

        if parent_idx in methyl_indices and heavy_neighbors:
            heavy_direction = unit(vec_sub(coords[heavy_neighbors[0]], parent_pos))
            side = perpendicular_unit(heavy_direction, normal)
            lift = unit(cross(heavy_direction, side))
            for i, h_idx in enumerate(h_indices):
                angle = 2 * math.pi * i / max(1, len(h_indices))
                h_direction = vec_add(
                    vec_scale(heavy_direction, -1.0 / 3.0),
                    vec_scale(
                        vec_add(
                            vec_scale(side, math.cos(angle)),
                            vec_scale(lift, math.sin(angle)),
                        ),
                        (8.0 / 9.0) ** 0.5,
                    ),
                )
                coords[h_idx] = vec_add(parent_pos, vec_scale(unit(h_direction), ch_length))
            continue

        if len(heavy_neighbors) >= 2:
            direction_sum = (0.0, 0.0, 0.0)
            for neighbor_idx in heavy_neighbors:
                direction_sum = vec_add(
                    direction_sum,
                    unit(vec_sub(coords[neighbor_idx], parent_pos)),
                )
            h_direction = unit(vec_scale(direction_sum, -1.0))
            coords[h_indices[0]] = vec_add(parent_pos, vec_scale(h_direction, ch_length))
            for h_idx in h_indices[1:]:
                coords[h_idx] = coords[h_indices[0]]
            continue

        if heavy_neighbors:
            base = unit(vec_sub(parent_pos, coords[heavy_neighbors[0]]))
        else:
            base = outward
        side = perpendicular_unit(base, normal)
        lift = unit(cross(base, side))
        for i, h_idx in enumerate(h_indices):
            angle = 2 * math.pi * i / max(1, len(h_indices))
            h_direction = unit(
                vec_add(
                    base,
                    vec_add(
                        vec_scale(side, math.cos(angle) * 0.65),
                        vec_scale(lift, math.sin(angle) * 0.65),
                    ),
                )
            )
            coords[h_idx] = vec_add(parent_pos, vec_scale(h_direction, ch_length))

    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in coords:
            coords[idx] = ligand_mid

    return coords


def ethylenediamine_ligand_positions(ligand_mol, donor_indices, target_sites):
    coords = {}

    n1_idx, n2_idx = donor_indices[:2]
    n1_pos, n2_pos = target_sites

    coords[n1_idx] = n1_pos
    coords[n2_idx] = n2_pos

    mid = (
        (n1_pos[0] + n2_pos[0]) / 2,
        (n1_pos[1] + n2_pos[1]) / 2,
        (n1_pos[2] + n2_pos[2]) / 2,
    )

    axis = unit(vec_sub(n2_pos, n1_pos))

    outward = unit(mid)
    if norm(outward) == 0:
        outward = (0.0, 0.0, 1.0)

    side = cross(axis, outward)
    if norm(side) == 0:
        side = (0.0, 0.0, 1.0)
    side = unit(side)

    n1_atom = ligand_mol.GetAtomWithIdx(n1_idx)
    n2_atom = ligand_mol.GetAtomWithIdx(n2_idx)

    c1_candidates = [
        nbr.GetIdx()
        for nbr in n1_atom.GetNeighbors()
        if nbr.GetSymbol() == "C"
    ]

    c2_candidates = [
        nbr.GetIdx()
        for nbr in n2_atom.GetNeighbors()
        if nbr.GetSymbol() == "C"
    ]

    if c1_candidates and c2_candidates:
        c1_idx = c1_candidates[0]
        c2_idx = c2_candidates[0]

        nc_length = 1.47
        cc_length = 1.53
        half_nn = norm(vec_sub(n2_pos, n1_pos)) / 2
        half_cc = cc_length / 2
        axis_offset = half_cc
        axial_nc_component = half_nn - axis_offset
        if abs(axial_nc_component) >= nc_length:
            axis_offset = max(0.2, half_nn - nc_length * 0.85)
            axial_nc_component = half_nn - axis_offset
            cc_length = 2 * axis_offset

        outward_offset = max(
            0.2,
            (nc_length * nc_length - axial_nc_component * axial_nc_component) ** 0.5,
        )

        coords[c1_idx] = vec_add(
            mid,
            vec_add(
                vec_scale(axis, -axis_offset),
                vec_scale(outward, outward_offset),
            ),
        )

        coords[c2_idx] = vec_add(
            mid,
            vec_add(
                vec_scale(axis, axis_offset),
                vec_scale(outward, outward_offset),
            ),
        )

        _place_methylene_hydrogens(coords, ligand_mol, c1_idx, [n1_idx, c2_idx])
        _place_methylene_hydrogens(coords, ligand_mol, c2_idx, [c1_idx, n2_idx])
        _place_two_tetrahedral_hydrogens(
            coords,
            ligand_mol,
            n1_idx,
            [(0.0, 0.0, 0.0), coords[c1_idx]],
            1.01,
        )
        _place_two_tetrahedral_hydrogens(
            coords,
            ligand_mol,
            n2_idx,
            [(0.0, 0.0, 0.0), coords[c2_idx]],
            1.01,
        )

    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()

        if idx in coords:
            continue

        if atom.GetSymbol() == "H":
            neighbors = atom.GetNeighbors()
            if neighbors:
                base_idx = neighbors[0].GetIdx()
                base = coords.get(base_idx, mid)

                h_out = unit(vec_sub(base, mid))
                if norm(h_out) == 0:
                    h_out = outward

                coords[idx] = vec_add(
                    base,
                    vec_add(
                        vec_scale(h_out, 0.8),
                        vec_scale(side, 0.45 if idx % 2 == 0 else -0.45),
                    ),
                )
            else:
                coords[idx] = mid
        else:
            coords[idx] = mid

    return coords


def triethyl_ligand_positions(ligand_mol, donor_idx, target, nh_distance=1.0, spread=0.8):
    direction = unit(target)

    metal_p_length = max(norm(target), 4.0)
    center = vec_scale(direction, metal_p_length)
    coords = {}
    coords[donor_idx] = center

    if abs(direction[0]) < 0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)

    u = unit(cross(direction, ref))
    v = unit(cross(direction, u))

    p_atom=ligand_mol.GetAtomWithIdx(donor_idx)

    first_carbons=[neighbor.GetIdx() for neighbor in p_atom.GetNeighbors() if neighbor.GetSymbol()=="C"]
    pc_length = 1.83
    cc_length = 1.53
    substituent_directions = _tetrahedral_substituent_directions(direction, u, v)

    for c_idx, pc_direction in zip(first_carbons[:3], substituent_directions):
        coords[c_idx] = vec_add(center, vec_scale(pc_direction, pc_length))

        second_carbons=[
            neighbor.GetIdx()
            for neighbor in ligand_mol.GetAtomWithIdx(c_idx).GetNeighbors()
            if neighbor.GetSymbol()=="C"
        ]
        if not second_carbons:
            _place_methyl_hydrogens(coords, ligand_mol, c_idx, donor_idx, direction)
            continue

        c2_idx=second_carbons[0]
        chain_side = _perpendicular_unit(pc_direction, direction)
        c1_to_c2_direction = unit(
            vec_add(
                vec_scale(pc_direction, 1.0 / 3.0),
                vec_scale(chain_side, (8.0 / 9.0) ** 0.5),
            )
        )
        coords[c2_idx]=vec_add(coords[c_idx], vec_scale(c1_to_c2_direction, cc_length))

        _place_methylene_hydrogens(coords, ligand_mol, c_idx, (donor_idx, c2_idx))
        _place_methyl_hydrogens(coords, ligand_mol, c2_idx, c_idx, pc_direction)

    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in coords:
            coords[idx] = center

    return coords

def terpyridine_ligand_positions(ligand_mol, donor_indices, target_sites):
    """Place tpy/terpy as a rigid planar ligand and let terminal M-N lengths adapt."""
    donor_indices = donor_indices[:3]
    target_sites = target_sites[:3]

    planar_mol = Chem.Mol(ligand_mol)
    rdDepictor.Compute2DCoords(planar_mol)
    conf = planar_mol.GetConformer()

    local_donors = []
    for donor_idx in donor_indices:
        pos = conf.GetAtomPosition(donor_idx)
        local_donors.append((pos.x, pos.y))

    left_donor, center_donor, right_donor = local_donors
    local_center = center_donor
    terminal_mid = (
        (left_donor[0] + right_donor[0]) / 2,
        (left_donor[1] + right_donor[1]) / 2,
    )
    local_axis = (
        right_donor[0] - left_donor[0],
        right_donor[1] - left_donor[1],
    )
    local_axis_length = (local_axis[0] ** 2 + local_axis[1] ** 2) ** 0.5
    local_out = (
        center_donor[0] - terminal_mid[0],
        center_donor[1] - terminal_mid[1],
    )
    local_out_length = (local_out[0] ** 2 + local_out[1] ** 2) ** 0.5
    if local_axis_length <= 1e-8 or local_out_length <= 1e-8:
        return translate_tridentate_ligand(
            ligand_mol,
            donor_indices,
            target_sites,
            reverse=False,
        )

    local_axis = (
        local_axis[0] / local_axis_length,
        local_axis[1] / local_axis_length,
    )
    local_out = (
        local_out[0] / local_out_length,
        local_out[1] / local_out_length,
    )

    target_center = target_sites[1]
    target_axis = unit(vec_sub(target_sites[2], target_sites[0]))
    target_out = unit(target_center)
    target_out = vec_sub(target_out, vec_scale(target_axis, dot(target_out, target_axis)))
    if norm(target_out) <= 1e-8:
        target_out = cross(target_axis, (0.0, 0.0, 1.0))
    if norm(target_out) <= 1e-8:
        target_out = cross(target_axis, (0.0, 1.0, 0.0))
    target_out = unit(target_out)

    heavy_bond_lengths = []
    for bond in planar_mol.GetBonds():
        begin_atom = planar_mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
        end_atom = planar_mol.GetAtomWithIdx(bond.GetEndAtomIdx())
        if "H" in {begin_atom.GetSymbol(), end_atom.GetSymbol()}:
            continue
        begin_pos = conf.GetAtomPosition(begin_atom.GetIdx())
        end_pos = conf.GetAtomPosition(end_atom.GetIdx())
        heavy_bond_lengths.append(
            math.dist((begin_pos.x, begin_pos.y), (end_pos.x, end_pos.y))
        )
    if heavy_bond_lengths:
        sorted_lengths = sorted(heavy_bond_lengths)
        median_length = sorted_lengths[len(sorted_lengths) // 2]
        scale = 1.39 / median_length
    else:
        scale = 1.0

    coords = {}
    for atom in planar_mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        centered = (pos.x - local_center[0], pos.y - local_center[1])
        axis_component = centered[0] * local_axis[0] + centered[1] * local_axis[1]
        out_component = centered[0] * local_out[0] + centered[1] * local_out[1]

        coords[idx] = vec_add(
            target_center,
            vec_add(
                vec_scale(target_axis, axis_component * scale),
                vec_scale(target_out, out_component * scale),
            ),
        )

    coords[donor_indices[1]] = target_center

    return coords

def bipyridine_ligand_positions(ligand_mol, donor_indices, target_sites, ligand_number=0):
    coords={}
    n1_idx, n2_idx = donor_indices[:2]
    n1_pos, n2_pos = target_sites

    coords[n1_idx] = n1_pos
    coords[n2_idx] = n2_pos


    n1_atom=ligand_mol.GetAtomWithIdx(n1_idx)
    n2_atom=ligand_mol.GetAtomWithIdx(n2_idx)
    axis=unit(vec_sub(n2_pos, n1_pos))
    mid = (
        (n1_pos[0] + n2_pos[0]) / 2,
        (n1_pos[1] + n2_pos[1]) / 2,
        (n1_pos[2] + n2_pos[2]) / 2,
    )
    outward=unit(mid)

    if norm(outward) == 0:
        outward = (0.0, 0.0, 1.0)

    side = cross(axis, outward)

    if norm(side) == 0:
        side = (0.0, 0.0, 1.0)

    side = unit(side)


    if ligand_number % 2 == 1:
        side = vec_scale(side, -1.0)

    ring_info = ligand_mol.GetRingInfo()
    rings = [list(ring) for ring in ring_info.AtomRings()]

    donor_rings = []
    for donor_idx in donor_indices:
        for ring in rings:
            if donor_idx in ring:
                donor_rings.append((donor_idx, ring))
                break

    ring_radius = 0.75

    for donor_idx, ring in donor_rings:
        donor_pos = coords[donor_idx]
        ring_center = vec_add(donor_pos, vec_scale(outward, ring_radius))

        ordered_ring = [donor_idx] + [idx for idx in ring if idx != donor_idx]

        angles = [
            math.pi,
            2 * math.pi / 3,
            math.pi / 3,
            0.0,
            -math.pi / 3,
            -2 * math.pi / 3,
        ]

        for idx, angle in zip(ordered_ring, angles):
            ring_pos = vec_add(
                vec_scale(outward, math.cos(angle) * ring_radius),
                vec_scale(side, math.sin(angle) * ring_radius),
            )
            coords[idx] = vec_add(ring_center, ring_pos)

    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()

        if idx in coords:
            continue

        if atom.GetSymbol() == "H":
            neighbors = atom.GetNeighbors()
            if neighbors:
                base_idx = neighbors[0].GetIdx()
                base = coords.get(base_idx, mid)

                h_out = unit(vec_sub(base, mid))

                if norm(h_out) == 0:
                    h_out = outward

                coords[idx] = vec_add(
                    base,
                        vec_scale(h_out, 0.45),
                )
            else:
                coords[idx] = mid
        else:
            coords[idx] = mid

    return coords


def planar_bidentate_aromatic_ligand_positions(ligand_mol, donor_indices, target_sites):
    """Place a rigid aromatic bidentate ligand in its N-metal-N plane."""
    n1_idx, n2_idx = donor_indices[:2]
    n1_pos, n2_pos = target_sites

    planar_mol = Chem.Mol(ligand_mol)
    rdDepictor.Compute2DCoords(planar_mol)
    conf = planar_mol.GetConformer()

    n1_2d = conf.GetAtomPosition(n1_idx)
    n2_2d = conf.GetAtomPosition(n2_idx)
    local_n1 = (n1_2d.x, n1_2d.y)
    local_n2 = (n2_2d.x, n2_2d.y)
    local_axis = (local_n2[0] - local_n1[0], local_n2[1] - local_n1[1])
    local_length = (local_axis[0] ** 2 + local_axis[1] ** 2) ** 0.5
    if local_length <= 1e-8:
        return translate_bidentate_ligand(ligand_mol, donor_indices, target_sites)

    local_axis = (local_axis[0] / local_length, local_axis[1] / local_length)
    local_perp = (-local_axis[1], local_axis[0])
    local_mid = (
        (local_n1[0] + local_n2[0]) / 2,
        (local_n1[1] + local_n2[1]) / 2,
    )
    centroid_perp = 0.0
    for atom in planar_mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        centered = (pos.x - local_mid[0], pos.y - local_mid[1])
        centroid_perp += centered[0] * local_perp[0] + centered[1] * local_perp[1]
    if centroid_perp < 0:
        local_perp = (-local_perp[0], -local_perp[1])

    target_axis = unit(vec_sub(n2_pos, n1_pos))
    n_mid = (
        (n1_pos[0] + n2_pos[0]) / 2,
        (n1_pos[1] + n2_pos[1]) / 2,
        (n1_pos[2] + n2_pos[2]) / 2,
    )
    target_out = unit(n_mid)
    if norm(target_out) <= 1e-8:
        target_out = (0.0, 0.0, 1.0)

    target_out = vec_sub(target_out, vec_scale(target_axis, dot(target_out, target_axis)))
    if norm(target_out) <= 1e-8:
        target_out = cross(target_axis, (0.0, 0.0, 1.0))
    if norm(target_out) <= 1e-8:
        target_out = cross(target_axis, (0.0, 1.0, 0.0))
    target_out = unit(target_out)

    target_length = norm(vec_sub(n2_pos, n1_pos))
    scale = target_length / local_length

    coords = {}
    for atom in planar_mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        centered = (pos.x - local_mid[0], pos.y - local_mid[1])
        axis_component = centered[0] * local_axis[0] + centered[1] * local_axis[1]
        perp_component = centered[0] * local_perp[0] + centered[1] * local_perp[1]

        placed = vec_add(
            n_mid,
            vec_add(
                vec_scale(target_axis, axis_component * scale),
                vec_scale(target_out, perp_component * scale),
            ),
        )
        coords[idx] = placed

    # Ensure the donor nitrogens are exactly on their metal-binding sites.
    coords[n1_idx] = n1_pos
    coords[n2_idx] = n2_pos
    return coords


def phenanthroline_ligand_positions(ligand_mol, donor_indices, target_sites):
    """Place phenanthroline as one rigid plane in the N-metal-N plane."""
    return planar_bidentate_aromatic_ligand_positions(
        ligand_mol,
        donor_indices,
        target_sites,
    )


def water_ligand_positions(ligand_mol, donor_idx, target):
    direction = unit(target)
    coords = {}

    coords[donor_idx] = target

    if abs(direction[0]) < 0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)

    side = unit(cross(direction, ref))

    h_indices = [
        atom.GetIdx()
        for atom in ligand_mol.GetAtoms()
        if atom.GetSymbol() == "H"
    ]

    hoh_angle = math.radians(104.5)
    oh_distance = 0.96
    forward = math.cos(hoh_angle / 2.0)
    lateral = math.sin(hoh_angle / 2.0)

    if len(h_indices) >= 2:
        coords[h_indices[0]] = vec_add(
            target,
            vec_scale(
                vec_add(
                    vec_scale(direction, forward),
                    vec_scale(side, lateral),
                ),
                oh_distance,
            ),
        )

        coords[h_indices[1]] = vec_add(
            target,
            vec_scale(
                vec_add(
                    vec_scale(direction, forward),
                    vec_scale(side, -lateral),
                ),
                oh_distance,
            ),
        )

    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in coords:
            coords[idx] = target

    return coords


def hydroxo_ligand_positions(ligand_mol, donor_idx, target):
    direction = unit(target)
    coords = {donor_idx: target}

    if abs(direction[0]) < 0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)

    side = unit(cross(direction, ref))
    oh_distance = 0.96
    h_direction = unit(
        vec_add(
            vec_scale(direction, 1.0 / 3.0),
            vec_scale(side, (8.0 / 9.0) ** 0.5),
        )
    )

    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetSymbol() == "H":
            coords[idx] = vec_add(target, vec_scale(h_direction, oh_distance))
        elif idx not in coords:
            coords[idx] = target

    return coords

def dmso_ligand_positions(ligand_mol, donor_idx, target, ligand_number=0):
    direction = unit(target)

    center = target
    coords = {}
    coords[donor_idx] = center

    if abs(direction[0]) < 0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)

    u = unit(cross(direction, ref))
    v = unit(cross(direction, u))

    angle_offset= ligand_number * math.pi
    u_rot = vec_add(
        vec_scale(u, math.cos(angle_offset)),
        vec_scale(v, math.sin(angle_offset)),
    )
    v_rot = vec_add(
        vec_scale(u, -math.sin(angle_offset)),
        vec_scale(v, math.cos(angle_offset)),
    )

    s_indices = [
        atom.GetIdx()
        for atom in ligand_mol.GetAtoms()
        if atom.GetSymbol() == "S"]
    o_indices = [
        atom.GetIdx()
        for atom in ligand_mol.GetAtoms()
        if atom.GetSymbol() == "O"]
    c_indices = [
        atom.GetIdx()
        for atom in ligand_mol.GetAtoms()
        if atom.GetSymbol() == "C"]
    s_idx=s_indices[0]
    o_idx=o_indices[0]
    so_length = 1.49
    sc_length = 1.79

    if donor_idx==s_idx:
        coords[s_idx]=target
        s_center=target
        substituent_directions = _tetrahedral_substituent_directions(
            direction,
            u_rot,
            v_rot,
        )

        o_pos=vec_add(s_center, vec_scale(substituent_directions[0], so_length))
        coords[o_idx]=o_pos

        for c_idx, sc_direction in zip(c_indices, substituent_directions[1:]):
            coords[c_idx] = vec_add(s_center, vec_scale(sc_direction, sc_length))
            _place_methyl_hydrogens(coords, ligand_mol, c_idx, s_idx, direction)

    elif donor_idx==o_idx:
        coords[o_idx]=target
        o_center=target
        s_direction = unit(
            vec_add(
                vec_scale(direction, 0.5),
                vec_scale(u_rot, (3.0 ** 0.5) / 2.0),
            )
        )
        s_pos=vec_add(o_center, vec_scale(s_direction, so_length))
        coords[s_idx]=s_pos
        s_side = _perpendicular_unit(s_direction, v_rot)
        s_lift = unit(cross(s_direction, s_side))
        carbon_directions = _tetrahedral_substituent_directions(
            s_direction,
            s_side,
            s_lift,
        )

        for c_idx, sc_direction in zip(c_indices, carbon_directions[:2]):
            coords[c_idx] = vec_add(s_pos, vec_scale(sc_direction, sc_length))
            _place_methyl_hydrogens(coords, ligand_mol, c_idx, s_idx, s_direction)

    for atom in ligand_mol.GetAtoms():
        idx= atom.GetIdx()
        if idx not in coords:
            coords[idx] = target
    return coords

def cyclopentadienyl_ligand_positions(ligand_mol,target,ligand_number=0):
    direction=unit(target)
    if abs(direction[0])<0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)

    u = unit(cross(direction, ref))
    v = unit(cross(direction, u))

    coords = {}
    metal_cp_length = max(norm(target), 3.0)
    ring_center=vec_scale(direction, metal_cp_length)

    c_indices = [
        atom.GetIdx()
        for atom in ligand_mol.GetAtoms()
        if atom.GetSymbol() == "C"]
    h_indices = [
        atom.GetIdx()
        for atom in ligand_mol.GetAtoms()
        if atom.GetSymbol() == "H"]
    
    angle_offset = ligand_number * math.pi / 5
    aromatic_cc_length = 1.40
    ch_length = 1.09
    ring_radius = aromatic_cc_length / (2.0 * math.sin(math.pi / 5.0))

    for i, c_idx in enumerate(c_indices[:5]):
        angle = angle_offset + 2 * math.pi * i / 5

        ring_pos = vec_add(
            vec_scale(u, math.cos(angle) * ring_radius),
            vec_scale(v, math.sin(angle) * ring_radius),
        )

        coords[c_idx] = vec_add(ring_center, ring_pos)

    for h_idx in h_indices:
        atom = ligand_mol.GetAtomWithIdx(h_idx)
        neighbors = atom.GetNeighbors()

        if neighbors:
            c_idx = neighbors[0].GetIdx()
            base = coords.get(c_idx, ring_center)

            outward = unit(vec_sub(base, ring_center))
            coords[h_idx] = vec_add(base, vec_scale(outward, ch_length))
        else:
            coords[h_idx] = ring_center

    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in coords:
            coords[idx] = ring_center

    return coords

def edta_ligand_positions(ligand_mol, donor_indices, target_sites):
    """
    Place all atoms of an EDTA-like hexadentate ligand around a metal at origin.

    donor_indices: 6 RDKit atom indices in canonical order
        (N1, O_a, O_b, N2, O_c, O_d) per LIGAND_DONOR_INDEX_OVERRIDES["EDTA"].
    target_sites: matching 3D coordinates (chosen by the dispatcher so that
        each N is cis to its 2 oxygens and to the other N).

    Strategy: for each carboxylate arm (M-N-CH2-C(=O)-O-M), place CH2 and the
    carbonyl C inside the M-N-O plane with a small outward radial bulge so the
    5-membered chelate ring closes visibly. The dangling =O extends further
    out along the carbonyl C radial. The ethylene bridge -CH2-CH2- sits on
    the N1-N2 mid-arc, also bulged outward.
    """
    coords = {}

    for d_idx, target in zip(donor_indices, target_sites):
        coords[d_idx] = target

    donor_set = set(donor_indices)
    n_donors = [
        idx for idx in donor_indices
        if ligand_mol.GetAtomWithIdx(idx).GetSymbol() == "N"
    ]
    o_donors = [
        idx for idx in donor_indices
        if ligand_mol.GetAtomWithIdx(idx).GetSymbol() == "O"
    ]


    arm = {}
    for o_idx in o_donors:
        o_atom = ligand_mol.GetAtomWithIdx(o_idx)
        carbonyl_c = next(
            (n.GetIdx() for n in o_atom.GetNeighbors() if n.GetSymbol() == "C"),
            None,
        )
        if carbonyl_c is None:
            continue
        c_atom = ligand_mol.GetAtomWithIdx(carbonyl_c)
        dangling_o = next(
            (n.GetIdx() for n in c_atom.GetNeighbors()
             if n.GetSymbol() == "O" and n.GetIdx() != o_idx),
            None,
        )
        ch2 = next(
            (n.GetIdx() for n in c_atom.GetNeighbors() if n.GetSymbol() == "C"),
            None,
        )
        parent_n = None
        if ch2 is not None:
            ch2_atom = ligand_mol.GetAtomWithIdx(ch2)
            parent_n = next(
                (n.GetIdx() for n in ch2_atom.GetNeighbors()
                 if n.GetSymbol() == "N" and n.GetIdx() in donor_set),
                None,
            )
        arm[o_idx] = (parent_n, ch2, carbonyl_c, dangling_o)

    def _place_chain_in_plane(start, end, plane_out, alphas, betas):
        # Place chain atoms in the plane (start, end, plane_out), at positions
        # mid + axis*alpha + outward*beta. axis runs start->end; outward is
        # plane_out re-orthogonalized against axis (so it lies in the plane and
        # is perpendicular to start->end).
        mid = vec_scale(vec_add(start, end), 0.5)
        axis = unit(vec_sub(end, start))
        out_proj = vec_scale(axis, dot(axis, plane_out))
        out = unit(vec_sub(plane_out, out_proj))
        return [
            vec_add(vec_add(mid, vec_scale(axis, a)), vec_scale(out, b))
            for a, b in zip(alphas, betas)
        ]

    ARM_ALPHAS = (-0.555, 0.955)
    ARM_BETAS = (1.196, 1.196)
    BRIDGE_ALPHAS = (-0.755, 0.755)
    BRIDGE_BETAS = (1.196, 1.196)
    DANGLING_LEN = 1.22

    for o_idx, (n_idx, ch2, c_idx, dangling) in arm.items():
        if n_idx is None or n_idx not in coords:
            continue
        n_pos = coords[n_idx]
        o_pos = coords[o_idx]
        plane_out = unit(vec_add(n_pos, o_pos))
        placed = _place_chain_in_plane(
            n_pos, o_pos, plane_out, ARM_ALPHAS, ARM_BETAS
        )

        if ch2 is not None and ch2 not in coords:
            coords[ch2] = placed[0]
        if c_idx is not None and c_idx not in coords:
            coords[c_idx] = placed[1]

        if dangling is not None and dangling not in coords and c_idx in coords:
            c_pos = coords[c_idx]
            radial = unit(c_pos)
            coords[dangling] = vec_add(c_pos, vec_scale(radial, DANGLING_LEN))

    # Ethylene bridge -CH2-CH2- between the two N donors.
    if len(n_donors) >= 2:
        n1_idx, n2_idx = n_donors[:2]
        bridge_pair = []
        for bond in ligand_mol.GetBonds():
            a = bond.GetBeginAtom()
            b = bond.GetEndAtom()
            if a.GetSymbol() != "C" or b.GetSymbol() != "C":
                continue
            a_ns = {n.GetIdx() for n in a.GetNeighbors()
                    if n.GetSymbol() == "N" and n.GetIdx() in donor_set}
            b_ns = {n.GetIdx() for n in b.GetNeighbors()
                    if n.GetSymbol() == "N" and n.GetIdx() in donor_set}
            if a_ns and b_ns and a_ns != b_ns:
                bridge_pair = [a.GetIdx(), b.GetIdx()]
                break

        if len(bridge_pair) == 2:
            n1_pos = coords[n1_idx]
            n2_pos = coords[n2_idx]
            plane_out = unit(vec_add(n1_pos, n2_pos))
            placed = _place_chain_in_plane(
                n1_pos, n2_pos, plane_out, BRIDGE_ALPHAS, BRIDGE_BETAS
            )
           
            for c_idx in bridge_pair:
                c_atom = ligand_mol.GetAtomWithIdx(c_idx)
                connects_n1 = any(
                    n.GetIdx() == n1_idx for n in c_atom.GetNeighbors()
                )
                coords[c_idx] = placed[0] if connects_n1 else placed[1]

   
    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in coords or atom.GetSymbol() == "H":
            continue
        anchor_pos = None
        for n in atom.GetNeighbors():
            if n.GetIdx() in coords:
                anchor_pos = coords[n.GetIdx()]
                break
        if anchor_pos is None:
            coords[idx] = (0.0, 0.0, 0.0)
            continue
        radial = unit(anchor_pos)
        coords[idx] = vec_add(anchor_pos, vec_scale(radial, 0.9))

  
    hydrogen_parents: dict[int, list[int]] = {}
    for atom in ligand_mol.GetAtoms():
        if atom.GetSymbol() != "H":
            continue
        neighbors = atom.GetNeighbors()
        if not neighbors:
            coords[atom.GetIdx()] = (0.0, 0.0, 0.0)
            continue
        hydrogen_parents.setdefault(neighbors[0].GetIdx(), []).append(atom.GetIdx())

    for parent_idx, h_indices in hydrogen_parents.items():
        if parent_idx not in coords:
            for h_idx in h_indices:
                coords[h_idx] = (0.0, 0.0, 0.0)
            continue

        heavy_neighbors = [
            neighbor.GetIdx()
            for neighbor in ligand_mol.GetAtomWithIdx(parent_idx).GetNeighbors()
            if neighbor.GetSymbol() != "H" and neighbor.GetIdx() in coords
        ]
        if len(h_indices) == 2 and len(heavy_neighbors) >= 2:
            _place_methylene_hydrogens(
                coords,
                ligand_mol,
                parent_idx,
                tuple(heavy_neighbors[:2]),
            )
            continue

        parent_pos = coords[parent_idx]
        radial = unit(parent_pos) if norm(parent_pos) > 0.1 else (0.0, 0.0, 1.0)
        for h_idx in h_indices:
            if h_idx not in coords:
                coords[h_idx] = vec_add(parent_pos, vec_scale(radial, 1.05))

    return coords


def build_complex_3d(
    parsed: ParsedComplex,
    distance: float = 2.0,
    geometry: str | None = None,
) -> Chem.Mol:

    if geometry is None:
        try:
            from coordchem.geometry import predict_geometry
            geometry = predict_geometry(parsed)
        except Exception:
            geometry = "octahedral"

    cn = parsed.coordination_number or 6
    sites = geometry_positions(geometry, cn, distance=distance)

    rw = Chem.RWMol()
    coords: dict[int, Position] = {}

    metal_atom = Chem.Atom(parsed.metal)
    metal_atom.SetNoImplicit(True)
    metal_idx = rw.AddAtom(metal_atom)
    coords[metal_idx] = (0.0, 0.0, 0.0)

    site_index = 0
    occupied=[False]*len(sites)
    tridentate_count=0
    def next_free_site():
        for i, is_occupied in enumerate(occupied):
            if not is_occupied:
                occupied[i] = True
                return i
        return None


    def next_free_pair(ligand_symbol: str, denticity: int):
        pairs = bidentate_site_pair_indices(
            geometry,
            len(sites),
            short_bidentate=is_short_bidentate_ligand(
                ligand_symbol,
                denticity,
            ),
        )

        for a, b in pairs:
            if not occupied[a] and not occupied[b]:
                occupied[a] = True
                occupied[b] = True
                return a, b

        return None


    def next_free_triplet(ligand_symbol: str | None = None):
        if "octahedral" in geometry.lower() and len(sites) >= 6:
            if ligand_symbol in {"tpy", "terpy"}:
                triplets = [(1, 4, 0), (3, 5, 2)]
            else:
                triplets = [(0, 2, 1), (3, 4, 5)]
        else:
            triplets = [(i, i + 1, i + 2) for i in range(0, len(sites) - 2, 3)]

        for a, b, c in triplets:
            if not occupied[a] and not occupied[b] and not occupied[c]:
                occupied[a] = True
                occupied[b] = True
                occupied[c] = True
                return a, b, c

        return None
    def next_free_quadruplet():
        if "octahedral" in geometry.lower() and len(sites)>=6:
            quadruplets=[(0,1,2,3)]
        else:
            quadruplets=[(i, i + 1, i + 2, i + 3)
            for i in range(0, len(sites) - 3, 4)]
        for a, b, c, d in quadruplets:
            if not occupied[a] and not occupied[b] and not occupied[c] and not occupied[d]:
                occupied[a] = True
                occupied[b] = True
                occupied[c] = True
                occupied[d] = True
                return a, b, c, d

        return None
    def next_free_sextuplet():
        if "octahedral" in geometry.lower() and len(sites)>=6:
            sextuplets=[(0,1,2,3,4,5)]
        else:
            sextuplets=[(i, i + 1, i + 2, i + 3,i+4,i+5)
            for i in range(0, len(sites) - 5, 6)]
        for a, b, c, d,e,f in sextuplets:
            if not occupied[a] and not occupied[b] and not occupied[c] and not occupied[d] and not occupied[e] and not occupied[f]:
                occupied[a] = True
                occupied[b] = True
                occupied[c] = True
                occupied[d] = True
                occupied[e] = True
                occupied[f] = True
                return a, b, c, d, e, f

        return None
    ligand_items = sorted(
    parsed.ligands.items(),
    key=lambda item: parsed.ligand_denticity.get(item[0], 1),
    reverse=True)
    
    for ligand_symbol, count in ligand_items:
        smiles = LIGAND_SMILES.get(ligand_symbol)

        if smiles is None:
            for _ in range(count):
                if site_index >= len(sites):
                    break

                placeholder = Chem.Atom("X")
                idx = rw.AddAtom(placeholder)
                coords[idx] = sites[site_index]
                rw.AddBond(metal_idx, idx, Chem.BondType.DATIVE)
                site_index += 1

            continue

        donor_symbol = parsed.donor_atoms.get(ligand_symbol, "?")
        donor_overrides = donor_index_overrides_for_ligand(
            ligand_symbol,
            donor_symbol,
        )
        denticity = parsed.ligand_denticity.get(ligand_symbol, 1)

        for _ in range(count):
            if site_index >= len(sites):
                break

            try:
                ligand_mol = build_ligand_3d(smiles)
            except ValueError:
                continue
            #bidentate case
            if denticity == 2 and len(donor_overrides) >= 2:
                bidentate_index = site_index // 2
                
                pair=next_free_pair(ligand_symbol, denticity)
                if pair is None:
                    break
                target_sites = (sites[pair[0]],sites[pair[1]])
                donor_indices = donor_overrides[:2]

                if ligand_symbol=="en":
                    local_coords=ethylenediamine_ligand_positions(ligand_mol, donor_indices,target_sites)

                elif ligand_symbol in ("bipy", "bpy"):
                    local_coords=planar_bidentate_aromatic_ligand_positions(ligand_mol, donor_indices,target_sites)
                elif ligand_symbol=="phen":
                    local_coords=phenanthroline_ligand_positions(ligand_mol, donor_indices,target_sites)
                elif ligand_symbol=="ox":
                    local_coords=oxalate_ligand_positions(ligand_mol, donor_indices,target_sites)
                elif ligand_symbol=="acac":
                    local_coords=acac_ligand_positions(ligand_mol, donor_indices,target_sites)
                else:
                    local_coords = translate_bidentate_ligand(
                    ligand_mol,
                    donor_indices,
                    target_sites,
                )

                offset = rw.GetNumAtoms()

                for atom in ligand_mol.GetAtoms():
                    global_idx = rw.AddAtom(atom)
                    coords[global_idx] = local_coords[atom.GetIdx()]

                for bond in ligand_mol.GetBonds():
                    a1 = bond.GetBeginAtomIdx() + offset
                    a2 = bond.GetEndAtomIdx() + offset
                    rw.AddBond(a1, a2, bond.GetBondType())

                for donor_idx in donor_indices:
                    rw.AddBond(
                        metal_idx,
                        donor_idx + offset,
                        Chem.BondType.DATIVE,
                    )

                site_index += 2
                continue
            #tridentate case
            elif denticity==3 and len(donor_overrides)>=3:
                triplet = next_free_triplet(ligand_symbol)
                if triplet is None:
                    break       
                target_sites = (sites[triplet[0]], sites[triplet[1]], sites[triplet[2]])
                tridentate_index = tridentate_count
                donor_indices = donor_overrides[:3]
                if ligand_symbol in ("tpy", "terpy"):
                    local_coords = terpyridine_ligand_positions(
                        ligand_mol,
                        donor_indices,
                        target_sites,
                    )
                else:
                    local_coords = translate_tridentate_ligand(
                        ligand_mol,
                        donor_indices,
                        target_sites,
                        reverse=(tridentate_index % 2 == 1),
                    )

                offset = rw.GetNumAtoms()

                for atom in ligand_mol.GetAtoms():
                    global_idx = rw.AddAtom(atom)
                    coords[global_idx] = local_coords[atom.GetIdx()]

                for bond in ligand_mol.GetBonds():
                    a1 = bond.GetBeginAtomIdx() + offset
                    a2 = bond.GetEndAtomIdx() + offset
                    rw.AddBond(a1, a2, bond.GetBondType())

                for donor_idx in donor_indices:
                    rw.AddBond(metal_idx, donor_idx + offset, Chem.BondType.DATIVE,)

                site_index += 3
                tridentate_count += 1
                continue
            #tetradentate case
            elif denticity==4 and len(donor_overrides)>=4:
                quadruplet=next_free_quadruplet()
                if quadruplet is None:
                    break       
                target_sites = (sites[quadruplet[0]], sites[quadruplet[1]],sites[quadruplet[2]],sites[quadruplet[3]])
                tridentate_index = 0
                donor_indices=donor_overrides[:4]

                local_coords=translate_tetradentate_ligand(ligand_mol,donor_indices,target_sites)
                offset=rw.GetNumAtoms()
                for atom in ligand_mol.GetAtoms():
                    global_idx = rw.AddAtom(atom)
                    coords[global_idx] = local_coords[atom.GetIdx()]

                for bond in ligand_mol.GetBonds():
                    a1 = bond.GetBeginAtomIdx() + offset
                    a2 = bond.GetEndAtomIdx() + offset
                    rw.AddBond(a1, a2, bond.GetBondType())

                for donor_idx in donor_indices:
                    rw.AddBond(metal_idx,donor_idx + offset,Chem.BondType.DATIVE)

                site_index += 4
                continue

            #pentadentate case
            elif ligand_symbol == "Cp" or denticity==5:
                    site=next_free_site()
                    if site is None:
                        break
                    target=sites[site]
                    if ligand_symbol == "Cp":

                        local_coords=cyclopentadienyl_ligand_positions(ligand_mol,target,site)
                    else:
                        ligand_conf=ligand_mol.GetConformer()
                        local_coords={}
                        for atom in ligand_mol.GetAtoms():
                            idx=atom.GetIdx()
                            old_pos=ligand_conf.GetAtomPosition(idx)
                            local_coords[idx]=(old_pos.x+target[0],old_pos.y+target[1],old_pos.z+target[2])
                    offset = rw.GetNumAtoms()

                    for atom in ligand_mol.GetAtoms():
                        global_idx = rw.AddAtom(atom)
                        coords[global_idx] = local_coords[atom.GetIdx()]

                    for bond in ligand_mol.GetBonds():
                        a1 = bond.GetBeginAtomIdx() + offset
                        a2 = bond.GetEndAtomIdx() + offset
                        rw.AddBond(a1, a2, bond.GetBondType())

                    site_index+=1
                    continue
            #else:
               # donor_idx_local = find_donor_atom(ligand_mol, donor_symbol)
               # rw.AddBond(
           # metal_idx, donor_idx_local + offset, Chem.BondType.DATIVE )
               # site_index+= 1
               # continue 
            elif denticity==6 and len(donor_overrides)>=6:

                if len(sites)<6:
                    break
                sextuplet = next_free_sextuplet()
                if sextuplet is None:
                    break

                
                donor_indices = donor_overrides[:6]
                if ligand_symbol in ("EDTA", "edta"):
                   # Donor order from LIGAND_DONOR_INDEX_OVERRIDES["EDTA"] = (N1, O, O, N2, O, O).
                   #  Both N donors are connected by a -CH2-CH2- bridge so they must be cis
                   # (sites[0] = +x, sites[2] = +y), and each N's two carboxylate O donors
                   # must also be cis to that N. The only trans pair below is O_b/O_d,
                   # one O from each N, which forms no chelate ring.
                   target_sites = (
                       sites[0],  # N1   -> +x
                       sites[3],  # O_a  -> -y  (cis to N1)
                       sites[5],  # O_b  -> -z  (cis to N1)
                       sites[2],  # N2   -> +y  (cis to N1)
                       sites[1],  # O_c  -> -x  (cis to N2)
                       sites[4],  # O_d  -> +z  (cis to N2)
                   )

                   local_coords=edta_ligand_positions(ligand_mol,donor_indices, target_sites)
                else:
                    target_sites = (sites[sextuplet[0]],sites[sextuplet[1]],sites[sextuplet[2]],sites[sextuplet[3]],sites[sextuplet[4]], sites[sextuplet[5]])
                    local_coords = translate_hexadentate_ligand(ligand_mol,donor_indices,target_sites)

                offset = rw.GetNumAtoms()

                for atom in ligand_mol.GetAtoms():
                    global_idx = rw.AddAtom(atom)
                    coords[global_idx] = local_coords[atom.GetIdx()]

                for bond in ligand_mol.GetBonds():
                    a1 = bond.GetBeginAtomIdx() + offset
                    a2 = bond.GetEndAtomIdx() + offset
                    rw.AddBond(a1, a2, bond.GetBondType())

                for donor_idx in donor_indices:
                    rw.AddBond(metal_idx,donor_idx + offset,Chem.BondType.DATIVE)

                site_index += 6
                continue

            primary_override = donor_overrides[0] if donor_overrides else None

            if primary_override is not None and primary_override < ligand_mol.GetNumAtoms():
                    donor_idx_local = primary_override
            else:
                try:
                    donor_idx_local = find_donor_atom(ligand_mol, donor_symbol)
                except ValueError:
                    continue
        
            #monodentate case
            site = next_free_site()
            if site is None:
                break
            target = sites[site]

            if ligand_symbol=="NH3":
                local_coords=ammonia_ligand_positions(ligand_mol, donor_idx_local,target)
            elif ligand_symbol=="py":
                local_coords=pyridine_ligand_positions(ligand_mol, donor_idx_local,target, site)    
            elif ligand_symbol=="CH3":
                local_coords=methyl_ligand_positions(ligand_mol, donor_idx_local,target, site)    
            elif ligand_symbol=="PMe3":
                local_coords=trimethyl_ligand_positions(ligand_mol, donor_idx_local,target)
            elif ligand_symbol=="PPh3":
                local_coords=triphenylphosphine_ligand_positions(ligand_mol, donor_idx_local,target)
            elif ligand_symbol=="dmso":
                local_coords=dmso_ligand_positions(ligand_mol, donor_idx_local,target,site)
            elif ligand_symbol=="PEt3":
                local_coords=triethyl_ligand_positions(ligand_mol,donor_idx_local,target)
            elif ligand_symbol in ("NO2","ONO"):
                local_coords=nitrito_ligand_positions(ligand_mol, donor_idx_local,target,ligand_symbol)
            elif ligand_symbol=="H2O":
                local_coords=water_ligand_positions(ligand_mol, donor_idx_local,target)
            elif ligand_symbol=="OH":
                local_coords=hydroxo_ligand_positions(ligand_mol, donor_idx_local,target)
            elif ligand_symbol in ("NCS","SCN"):
                local_coords=thiocyanato_ligand_positions(ligand_mol, donor_idx_local,target,ligand_symbol)    
            elif ligand_symbol=="N3":
                local_coords=azido_ligand_positions(ligand_mol, donor_idx_local,target)
            elif ligand_mol.GetNumAtoms() == 2:
                local_coords = diatomic_ligand_positions(
                    ligand_mol,
                    donor_idx_local,
                    target,
                    bond_length=1.2,
                )
            else:
                ligand_conf = ligand_mol.GetConformer()
                donor_pos = ligand_conf.GetAtomPosition(donor_idx_local)

                local_coords = {}

                for atom in ligand_mol.GetAtoms():
                    old_pos = ligand_conf.GetAtomPosition(atom.GetIdx())
                    local_coords[atom.GetIdx()] = (
                        old_pos.x - donor_pos.x + target[0],
                        old_pos.y - donor_pos.y + target[1],
                        old_pos.z - donor_pos.z + target[2],
                    )

            offset = rw.GetNumAtoms()

            for atom in ligand_mol.GetAtoms():
                global_idx = rw.AddAtom(atom)
                coords[global_idx] = local_coords[atom.GetIdx()]

            for bond in ligand_mol.GetBonds():
                a1 = bond.GetBeginAtomIdx() + offset
                a2 = bond.GetEndAtomIdx() + offset
                rw.AddBond(a1, a2, bond.GetBondType())

            rw.AddBond(
                metal_idx,
                donor_idx_local + offset,
                Chem.BondType.DATIVE,
            )

            site_index += 1

    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())

    for atom_idx, position in coords.items():
        conf.SetAtomPosition(atom_idx, position)
    mol.AddConformer(conf, assignId=True)
    return mol

   



#functions to display diatomic ligands correctly
def normalize(v):
    length=(v[0]**2+v[1]**2+v[2]**2)**0.5
    if length==0:
        return(1.0,0.0,0.0)
    return (v[0]/length,v[1]/length,v[2]/length)


def diatomic_ligand_positions(ligand_mol, donor_idx, target, bond_length=1.2):
    direction = normalize(target)
    coords = {}
    for atom in ligand_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx == donor_idx:
            coords[idx] = target
        else:
            coords[idx] = (
                target[0] + direction[0] * bond_length,
                target[1] + direction[1] * bond_length,
                target[2] + direction[2] * bond_length,
            )
    return coords


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def parse_complex_input(complex_input: str | ParsedComplex) -> ParsedComplex:
    """Accept a formula string, compound name, or already parsed complex."""
    if isinstance(complex_input, ParsedComplex):
        return complex_input
    if isinstance(complex_input, str):
        try:
            return parse_formula(complex_input)
        except Exception:
            return parse_name(complex_input)
    parsed = getattr(complex_input, "parsed", None)
    if isinstance(parsed, ParsedComplex):
        return parsed
    raise TypeError(
        "Expected a ParsedComplex, Complex, or formula/name string"
    )


def _to_parsed(complex_or_formula) -> ParsedComplex:
    """Best-effort coercion to a ``ParsedComplex``."""
    return parse_complex_input(complex_or_formula)


def _cp_ring_centers(mol: Chem.Mol) -> list[Position]:
    conf = mol.GetConformer()
    centers = []

    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 5:
            continue
        if any(mol.GetAtomWithIdx(idx).GetSymbol() != "C" for idx in ring):
            continue

        centers.append(
            (
                sum(conf.GetAtomPosition(idx).x for idx in ring) / 5.0,
                sum(conf.GetAtomPosition(idx).y for idx in ring) / 5.0,
                sum(conf.GetAtomPosition(idx).z for idx in ring) / 5.0,
            )
        )

    if centers:
        return centers

    adjacency = {idx: set() for idx in range(1, mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if 0 in {begin_idx, end_idx}:
            continue
        adjacency[begin_idx].add(end_idx)
        adjacency[end_idx].add(begin_idx)

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
            for neighbor_idx in adjacency[idx]:
                if neighbor_idx not in seen:
                    seen.add(neighbor_idx)
                    stack.append(neighbor_idx)

        carbon_indices = [
            idx for idx in component if mol.GetAtomWithIdx(idx).GetSymbol() == "C"
        ]
        heavy_symbols = [
            mol.GetAtomWithIdx(idx).GetSymbol()
            for idx in component
            if mol.GetAtomWithIdx(idx).GetSymbol() != "H"
        ]
        if len(carbon_indices) != 5 or any(symbol != "C" for symbol in heavy_symbols):
            continue

        centers.append(
            (
                sum(conf.GetAtomPosition(idx).x for idx in carbon_indices) / 5.0,
                sum(conf.GetAtomPosition(idx).y for idx in carbon_indices) / 5.0,
                sum(conf.GetAtomPosition(idx).z for idx in carbon_indices) / 5.0,
            )
        )

    return centers


def _add_dashed_line_to_view(view, start: Position, end: Position) -> None:
    dash_count = 7
    gap_fraction = 0.45
    vector = vec_sub(end, start)

    for i in range(dash_count):
        dash_start_t = i / dash_count
        dash_end_t = (i + (1.0 - gap_fraction)) / dash_count
        dash_start = vec_add(start, vec_scale(vector, dash_start_t))
        dash_end = vec_add(start, vec_scale(vector, dash_end_t))
        view.addCylinder(
            {
                "start": {
                    "x": dash_start[0],
                    "y": dash_start[1],
                    "z": dash_start[2],
                },
                "end": {
                    "x": dash_end[0],
                    "y": dash_end[1],
                    "z": dash_end[2],
                },
                "radius": 0.035,
                "color": "grey",
                "fromCap": 1,
                "toCap": 1,
            }
        )


def _add_cp_center_lines_to_view(view, mol: Chem.Mol) -> None:
    if mol.GetNumConformers() == 0:
        return

    metal_pos = mol.GetConformer().GetAtomPosition(0)
    metal = (metal_pos.x, metal_pos.y, metal_pos.z)
    for center in _cp_ring_centers(mol):
        _add_dashed_line_to_view(view, metal, center)


#function to display the molecule on a notebook

def view_complex_3d(
    complex_or_formula,
    width: int = 400,
    height: int = 400,
    distance: float = 2.0,
    geometry: str | None = None,
):
    import py3Dmol 

    parsed = _to_parsed(complex_or_formula)
    mol = build_complex_3d(parsed, distance=distance, geometry=geometry)

    block = Chem.MolToMolBlock(mol, kekulize=False) #pour ne pas forcer la conversion des cycles aromatiques
    view = py3Dmol.view(width=width, height=height)
    view.removeAllModels()
    view.addModel(block, "sdf")
    view.setStyle({}, {"stick": {}, "sphere": {"scale": 0.25}})
    view.setStyle({}, {"stick": {"radius": 0.12}, "sphere": {"scale": 0.18}})

    if "Cp" in parsed.ligands:
        _add_cp_center_lines_to_view(view, mol)


    view.setBackgroundColor("white")
    view.zoomTo()
    return view

#useful function to display the complex on an app such as streamlit
def complex_3d_html(
    complex_or_formula,
    width: int = 400,
    height: int = 400,
    distance: float = 2.0,
    geometry: str | None = None,
) -> str:
    view = view_complex_3d(
        complex_or_formula,
        width=width,
        height=height,
        distance=distance,
        geometry=geometry,
    )
    return view._make_html()
