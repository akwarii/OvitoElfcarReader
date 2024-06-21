import warnings
from typing import Any

import numpy as np
from pymatgen.io.vasp import Elfcar

warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

from ovito.io import FileReaderInterface
from ovito.data import DataCollection

class ElfcarReader(FileReaderInterface):

    @staticmethod
    def detect(filename: str):
        is_elfcar = "ELFCAR" in filename
        print(is_elfcar)
        return is_elfcar

    def parse(self, data: DataCollection, filename: str, **kwargs: Any):
        elfcar = Elfcar.from_file(filename)
        struct = elfcar.structure
        elf = elfcar.data['total']
        
        # Create cell
        cell = data.create_cell(
            np.hstack((struct.lattice.matrix, np.zeros((3,1)))),
            pbc=(True, True, True)
        )
        
        # Get particle count
        particle_count = len(struct)
        
        # Create particle container
        particles = data.create_particles(count=particle_count)
        
        # Populate particle properties
        type_property = particles.create_property(
            "Particle Type", data=struct.atomic_numbers
        )
        position_property = particles.create_property(
            "Position", data=struct.cart_coords
        )
        
        # Create voxel grid
        grid = data.grids.create(
            identifier="Electron localization function",
            domain=cell,
            shape=elf.shape,
        )
        
        # Populate voxel grid properties
        elf_property = grid.create_property(
            "Electron localization function",
            data=elf.flatten(order="F"),
        )
