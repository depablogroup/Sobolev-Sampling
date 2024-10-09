#!/usr/bin/env python3
# Script to run alanine tripeptide in PySAGES
from pysages.methods import Sirens
from pysages.colvars import DihedralAngle
from pysages.utils import try_import
# %%
openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")
from openmm import *

from openmm.unit import *

import math
import numpy
import pysages
import dill as pickle

import matplotlib.pyplot as plt

# %%
pi = numpy.pi
kB = 0.008314462618
# %%
def generate_simulation(T=298.15 * kelvin, dt=1.0 * femtoseconds):
    print("Loading AMBER files...")
    pdb = app.PDBFile('input.pdb')
    ff = app.ForceField("amber99sb.xml")
    topology=pdb.topology
    system = ff.createSystem(topology,
        nonbondedMethod=app.PME,
        switchDistance=1.0 * nanometer,
        nonbondedCutoff=1.2 * nanometer,
        constraints=app.HBonds,
    )
    # Create the integrator to do Langevin dynamics
    integrator = openmm.LangevinIntegrator(
        300 * kelvin,  # Temperature of heat bath
        1.0 / picoseconds,  # Friction coefficient
        1.0 * femtoseconds,  # Time step
    )

    # Define the platform to use; CUDA, OpenCL, CPU, or Reference. Or do not specify
    platform = Platform.getPlatformByName("CUDA")
    # Create the Simulation object
    sim = app.Simulation(topology, system, integrator, platform)

    # Set the particle positions
    sim.context.setPositions(pdb.getPositions(frame=0))
    sim.reporters.append(app.PDBReporter("output.pdb", 100000))
    sim.reporters.append(app.DCDReporter("output.dcd", 200000))
    sim.reporters.append(app.StateDataReporter("data.txt", 10000, step=True, potentialEnergy=True, totalEnergy=True,separator=" "))

    # Minimize the energy
    sim.minimizeEnergy()
    return sim
# Collective variable
cvs = (
    DihedralAngle([4,6,8,14]),
    DihedralAngle([14,16,18,24]),
    DihedralAngle([24,26,28,34]),
)

grid = pysages.Grid(lower=(-numpy.pi,-numpy.pi,-numpy.pi), upper=(numpy.pi,numpy.pi,numpy.pi), shape=(11,11,11), periodic=True)
timesteps = 5000000
topology=(14,4)
kT=kB*300.0
method = Sirens(cvs, grid,topology=topology,kT=kT, mode='cff')
state = pysages.run(methodo, generate_simulation, timesteps)
pysages.save(state,'restart'+str(windows)+'.pkl')
result = pysages.analyze(state)
mesh = numpy.asarray(result["mesh"])
energy = numpy.asarray(result["free_energy"])
forces = numpy.asarray(result["mean_force"])
histogram = numpy.asarray(result["histogram"])
numpy.savetxt("FES-5ns"+str(windows)+".csv", numpy.hstack([mesh, energy.reshape(-1,1)]))
numpy.save("FES-5ns"+str(windows)+".npy",numpy.hstack([mesh, energy.reshape(-1,1)]))
numpy.savetxt("Force-5ns"+str(windows)+".csv", numpy.hstack([mesh, forces.reshape(-1,3)]))
numpy.save("Force-5ns"+str(windows)+".npy",numpy.hstack([mesh, forces.reshape(-1,3)]))
numpy.savetxt("Histogram-5ns"+str(windows)+".csv", numpy.hstack([mesh, histogram.reshape(-1,1)]))
numpy.save("Histogram-5ns"+str(windows)+".npy",numpy.hstack([mesh, histogram.reshape(-1,1)]))
