# %%
from pysages.backends import SamplingContext
from pysages.colvars.core import CollectiveVariable
from pysages.methods import SpectralABF
from pysages.utils import try_import
from pysages.methods import CVRestraints
import time
import dill as pickle
import numpy
import pysages
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.numpy import linalg as linalg
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
    prmtop = app.AmberPrmtopFile('system.prmtop')
    inpcrd = app.AmberInpcrdFile('last.rst')
    system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer,
        constraints=app.HBonds)
    integrator = openmm.LangevinIntegrator(T, 1 / unit.picosecond, dt)
    integrator.setRandomNumberSeed(42)
    platform = openmm.Platform.getPlatformByName('CUDA')
    simulation = app.Simulation(prmtop.topology, system, integrator,platform)
    simulation.context.setPositions(inpcrd.positions)
    simulation.minimizeEnergy()
    simulation.reporters.append(
        app.StateDataReporter("log.dat", 10000, step=True, potentialEnergy=True, temperature=True)
    )
    return simulation
# %%
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
# %% Collective variables in PySAGES
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def periodic(distance,box):
    return jnp.mod(distance + box * 0.5, box) - 0.5 * box
class Distance_periodic(CollectiveVariable):
    def __init__(self,indices,box=[1.0,1.0,1.0]):
        super().__init__(indices,2)
        self.box=jnp.array(box)
    @property
    def function(self):
        return lambda r1, r2: distanceperiodic(r1, r2, self.box)
def distanceperiodic(p1, p2,box):
    return linalg.norm(periodic(p1-p2,box))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%
def run_example(timesteps):
    box = [4.026354, 4.026354, 4.026354]
    cvs = Distance_periodic([[5133],[1882]],box)
    grid = pysages.Grid(lower=(0.6,), upper=(1.0,), shape=(128,), periodic=False)
    restraints=CVRestraints(lower=(0.5,), upper=(1.1,), kl=(10000.0,), ku=(10000.0,))
    # method
    method = SpectralABF(cvs, grid,restraints=restraints)
    tic = time.perf_counter()
    sampling_context = SamplingContext(method, generate_simulation)
    state = pysages.run(sampling_context, timesteps)
    with open("restart1.pickle", "wb") as f:
        pickle.dump(state, f)
    toc = time.perf_counter()
    print(f"Simulation completerd in {toc - tic:0.4f} seconds.")

    return pysages.analyze(state)


# %%
result = run_example(int(4e7))
plot_energy(result)
plot_histogram(result)
plot_forces(result)
save_energy_forces(result)
