import warnings
from typing import Any

from pymatgen.io.vasp import Elfcar
from pymatgen.core import Structure

warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")

from ovito.io import FileReaderInterface
from ovito.data import DataCollection
from ovito.vis import VoxelGridVis


def pymatgen_to_ovito(
    struct: Structure, data: DataCollection | None = None
) -> DataCollection:
    if not data:
        data = DataCollection()

    # Set the unit cell and origin
    cell = data.create_cell(matrix=struct.lattice.matrix.T, pbc=struct.lattice.pbc)
    cell[:, 3] = [0.0, 0.0, 0.0]

    # Create particle property from atomic positions
    particles = data.create_particles(count=len(struct))
    particles.create_property("Position", data=struct.cart_coords)

    # Create named particle types from chemical symbols
    types = particles.create_property("Particle Type")
    symbols = [s.specie.symbol for s in struct.sites]

    # Map chemical element names to numeric type IDs.
    for i, sym in enumerate(symbols):
        types[i] = types.add_type_name(sym, particles).id

    return data


class ElfcarReader(FileReaderInterface):

    @staticmethod
    def detect(filename: str):
        is_elfcar = "ELFCAR" in filename
        return is_elfcar

    def parse(self, data: DataCollection, filename: str, **kwargs: Any):
        elfcar = Elfcar.from_file(filename)
        struct = elfcar.structure
        elf = elfcar.data["total"]

        data = pymatgen_to_ovito(struct, data)

        # Create voxel grid
        grid = data.grids.create(
            identifier="Electron localization function",
            domain=data.cell,
            shape=elf.shape,
            vis=VoxelGridVis(
                enabled=False,
                color_mapping_interval=(0.0, 1.0),
                highlight_grid_lines=False,
                interpolate_colors=True,
            ),
        )

        # Populate voxel grid properties
        elf_property = grid.create_property(
            "Electron localization function",
            data=elf.flatten(order="F"),
        )

        # Maps the grid to the ELF property
        grid.vis.color_mapping_property = "Electron localization function"
