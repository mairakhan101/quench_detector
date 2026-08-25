"""
hts_tape_env — HTS-tape thermal-runaway physics, a Gymnasium quench-detection
environment, and a surrogate-training pipeline.

Import layers
-------------
    from hts_tape_env import physics          # analytic solver, REBCO (source of truth)
    from hts_tape_env import physics_bi2223   # separate analytic solver, Bi-2223 (see its docstring)
    from hts_tape_env import thermal          # adiabatic time-margin / heat capacity (REBCO/Cu-SS only)
    from hts_tape_env.env import QuenchDetectorEnv
    from hts_tape_env.datagen import generate_dataset
    from hts_tape_env.surrogates import SPECS, train_spec, load_surrogate

    # Trajectory (sequence-model) pipeline -- rolls out the gym env itself
    # instead of sweeping the solver, so surrogates train on the same noisy,
    # ramped telemetry a deployed policy would see:
    from hts_tape_env.rollout import generate_rollout_dataset, generate_grid_dataset
    from hts_tape_env.seq_surrogates import train_seq_surrogate, load_seq_surrogate

    # FEM ground truth (//eagle/fanc_ldrd/magnets_fem/sweep.h5): real
    # RTD + voltage-tap telemetry from a 3D electrothermal solve, as a
    # tabular loader and as a drop-in replay Gym env with the real sensor
    # layout as its observation (see each module's docstring for caveats):
    from hts_tape_env.fem_data import load_fem_sweep, load_fem_positions
    from hts_tape_env.fem_env import FemReplayQuenchEnv
"""

__all__ = ["physics", "thermal"]
