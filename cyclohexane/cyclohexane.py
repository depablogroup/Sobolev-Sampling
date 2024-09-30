# %%
from pysages.colvars.core import ThreePointCV
from pysages.colvars.angles import dihedral_angle
from pysages.methods import SpectralABF
from pysages.utils import try_import
import matplotlib.pyplot as plt
import time

import numpy
import pysages

openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")


# %%
pi = numpy.pi
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

T = 300.0 * unit.kelvin


# %%
def generate_simulation(
    pdb_filename="input.pdb", T=T, dt=1.0 * unit.femtoseconds
):
    prmtop = app.AmberPrmtopFile('in.prmtop')
    inpcrd = app.AmberInpcrdFile('in.inpcrd')
    pdb = app.PDBFile(pdb_filename)
    system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer,
        constraints=app.HBonds)
    integrator = openmm.LangevinIntegrator(T, 1 / unit.picosecond, dt)
    integrator.setRandomNumberSeed(42)

    platform = openmm.Platform.getPlatformByName('CUDA')
    simulation = app.Simulation(prmtop.topology, system, integrator,platform)
    simulation.context.setPositions(inpcrd.positions)
    simulation.minimizeEnergy()

    return simulation


# %%
# %%
class ThreeDihedralsMean(ThreePointCV):
    @property
    def function(self):
        return three_dihedrals_mean

def three_dihedrals_mean(g1, g2, g3):
    d1 = dihedral_angle(*g1)
    d2 = dihedral_angle(*g2)
    d3 = dihedral_angle(*g3)
    return (d1 + d2 + d3) / 3

def plot_energy(result):
    fig, ax = plt.subplots()

    ax.set_xlabel("CV")
    ax.set_ylabel("Free energy $[\\epsilon]$")

    free_energy = numpy.asarray(result["free_energy"])
    x = numpy.asarray(result["mesh"])
    ax.plot(x, free_energy, color="teal")

    fig.savefig("energy.png")


def plot_forces(result):
    fig, ax = plt.subplots()

    ax.set_xlabel("CV")
    ax.set_ylabel("Forces $[\\epsilon]$")

    forces = numpy.asarray(result["mean_force"])
    x = numpy.asarray(result["mesh"])
    ax.plot(x, forces, color="teal")

    fig.savefig("forces.png")


def plot_histogram(result):
    fig, ax = plt.subplots()

    ax.set_xlabel("CV")
    ax.set_ylabel("Histogram $[\\epsilon]$")

    hist = numpy.asarray(result["histogram"])
    x = numpy.asarray(result["mesh"])
    ax.plot(x, hist, color="teal")

    fig.savefig("histogram2.png")


def save_energy_forces(result):
    Energy = numpy.asarray(result["free_energy"])
    Forces = numpy.asarray(result["mean_force"])
    Grid = numpy.asarray(result["mesh"])
    hist = numpy.asarray(result["histogram"])
    numpy.savetxt("FES.csv", numpy.column_stack([Grid, Energy]))
    numpy.savetxt("Forces.csv", numpy.column_stack([Grid, Forces]))
    numpy.savetxt("Histogram.csv", numpy.column_stack([Grid, hist]))
# %%
def run_example(timesteps):
    cvs = (
        ThreeDihedralsMean(([0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 0, 1])),
    )

    grid = Grid(lower=(-pi / 3,), upper=(pi / 3,), shape=(256,),periodic=True)
    # method
    method = SpectralABF(cvs, grid)
    tic = time.perf_counter()
    run_result = pysages.run(method, generate_simulation, timesteps)
    toc = time.perf_counter()
    print(f"Simulation completerd in {toc - tic:0.4f} seconds.")
    return pysages.analyze(run_result)
# %%
result = run_example(int(1e6))
plot_energy(result)
plot_forces(result)
plot_histogram(result)
save_energy_forces(result)
