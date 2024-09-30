#!/usr/bin/env python3
# %% Script to run WT-MetaD in plumed
from openmm import *
from openmm.app import *
from openmm.unit import *
from openmmplumed import PlumedForce
# %%
prmtop = AmberPrmtopFile('in.prmtop')
inpcrd = AmberInpcrdFile('in.inpcrd')
system = prmtop.createSystem(
    nonbondedMethod=PME,
    rigidWater=True,
    switchDistance=1.0 * nanometer,
    nonbondedCutoff=1.2 * nanometer,
    constraints=HBonds,
)
    # Create the integrator to do Langevin dynamics
integrator = openmm.LangevinIntegrator(
    300 * kelvin,  # Temperature of heat bath
    1.0 / picoseconds,  # Friction coefficient
    1.0 * femtoseconds,  # Time step
)
    # Plumed parameters
script = """
d1: TORSION ATOMS=1,11,9,8
d2: TORSION ATOMS=1,10,8,9
uwall: UPPER_WALLS ARG=d1,d2 AT=1.134464014,1.134464014 KAPPA=550.0,550.0 EXP=2,2 EPS=1,1 OFFSET=0,0
lwall: LOWER_WALLS ARG=d1,d2 AT=-1.134464014,-1.134464014 KAPPA=550.0,550.0 EXP=2,2 EPS=1,1 OFFSET=0,0
zx: COMBINE ARG=d1,d2 COEFFICIENTS=-0.6180339887498949,-0.6180339887498949 POWERS=1,1 PERIODIC=-1.0471975511965976,1.0471975511965976
zy: COMBINE ARG=d1,d2 COEFFICIENTS=0.8506508083520398,-0.8506508083520398 POWERS=1,1 PERIODIC=-1.0471975511965976,1.0471975511965976
eabf: METAD ARG=zx,zy SIGMA=0.32,0.32 HEIGHT=1.2 PACE=500 BIASFACTOR=5 TEMP=300.0
PRINT STRIDE=500 ARG=* FILE=COLVAR"""
system.addForce(PlumedForce(script))
    # Define the platform to use; CUDA, OpenCL, CPU, or Reference. Or do not specify
platform = Platform.getPlatformByName("CUDA")
    # Create the Simulation object

sim = Simulation(prmtop.topology, system, integrator, platform)

    # Set the particle positions
sim.context.setPositions(inpcrd.positions)
sim.reporters.append(PDBReporter("output.pdb", 5000000))
sim.reporters.append(DCDReporter("output.dcd", 5000000))
sim.reporters.append(StateDataReporter("data.txt", 20000, step=True, potentialEnergy=True,separator=" "))

    # Minimize the energy
sim.minimizeEnergy()
sim.step(40000000)

