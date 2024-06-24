from src.OvitoElfcarReader import pymatgen_to_ovito
from pymatgen.core import Structure


def test_pymatgen_to_ovito():
    struct = Structure.from_file('tests/data/POSCAR')
    pymatgen_to_ovito(struct)

if __name__ == '__main__':
    test_pymatgen_to_ovito()