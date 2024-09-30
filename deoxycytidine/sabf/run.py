# %%
from pysages.backends import SamplingContext
from pysages.colvars.core import TwoPointCV
from pysages.colvars.angles import dihedral_angle
from pysages.methods import SpectralABF
from pysages.utils import try_import
from pysages.methods import CVRestraints
import time
import dill as pickle
import numpy
import pysages
import matplotlib.pyplot as plt
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
# %% Collective variables in PySAGES
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class TwoDihedralsMean(TwoPointCV):
    @property
    def function(self):
        return two_dihedrals_mean

def two_dihedrals_mean(g1, g2):
    d1 = dihedral_angle(*g1)
    d2 = dihedral_angle(*g2)
    return (d1 + d2) / (2.0*np.cos(4*np.pi/5.0))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class TwoDihedralsDiff(TwoPointCV):
    @property
    def function(self):
        return two_dihedrals_diff

def two_dihedrals_diff(g1, g2):
    d1 = dihedral_angle(*g1)
    d2 = dihedral_angle(*g2)
    return (d1 - d2) / (2.0*np.sin(4*np.pi/5.0))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_energy(result):
    fig, ax = plt.subplots()

    ax.set_xlabel("CV")
    ax.set_ylabel("Free energy $[\\epsilon]$")

    A = numpy.asarray(result["free_energy"])
    x = numpy.asarray(result["mesh"])
    surface = (A.max() - A).reshape(32,32)

    fig, ax = plt.subplots()
    im = ax.imshow(
        surface,
        interpolation="bicubic",
        origin="lower",
        extent=[-pi/3.0, pi/3.0, -pi/3.0, pi/3.0],
        aspect=1,
    )
    ax.contour(
        surface, levels=15, linewidths=0.75, colors="k", extent=[-pi/3.0, pi/3.0, -pi/3.0, pi/3.0]
    )
    plt.colorbar(im)
    fig.savefig("energy.png")

def plot_histogram(result):
    fig, ax = plt.subplots()

    ax.set_xlabel("CV")
    ax.set_ylabel("Histogram $[\\epsilon]$")

    A = numpy.asarray(result["histogram"])
    x = numpy.asarray(result["mesh"])
    surface = A.reshape(32,32)

    fig, ax = plt.subplots()
    im = ax.imshow(
        surface,
        interpolation="bicubic",
        origin="lower",
        extent=[-pi/3.0, pi/3.0, -pi/3.0, pi/3.0],
        aspect=1,
    )
    ax.contour(
        surface, levels=15, linewidths=0.75, colors="k", extent=[-pi/3.0, pi/3.0, -pi/3.0, pi/3.0]
    )
    plt.colorbar(im)
    fig.savefig("histogram.png")

def save_energy_forces(result):
    Energy = numpy.asarray(result["free_energy"])
    Forces = numpy.asarray(result["mean_force"])
    Grid = numpy.asarray(result["mesh"])
    hist = numpy.asarray(result["histogram"])
    numpy.savetxt("FES.csv", numpy.hstack([Grid, Energy.reshape(-1,1)]))
    numpy.savetxt("Forces.csv", numpy.hstack([Grid, Energy.reshape(-1,2)]))
    numpy.savetxt("Histogram.csv", numpy.hstack([Grid, hist.reshape(-1,1)]))
# %%
def run_example(timesteps):
    dih1=[0,10,8,7]
    dih2=[0,9,7,8]
    cvs = (
    TwoDihedralsMean([dih1,dih2]),TwoDihedralsDiff([dih1,dih2])
)
    grid = pysages.Grid(lower=(-pi/3.0,-pi/3.0), upper=(pi/3.0,pi/3.0), shape=(32,32), periodic=False)
    kT = (kB * T).value_in_unit(unit.kilojoules_per_mole)
    restraints=CVRestraints(lower=(-pi/3.0,-pi/3.0), upper=(pi/3.0,pi/3.0), kl=(1000.0,1000.0), ku=(1000.0,1000.0))
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
save_energy_forces(result)
