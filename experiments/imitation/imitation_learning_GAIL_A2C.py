import argparse
import os
import pickle
from imitation.data.rollout import flatten_trajectories
from imitation.util import util

import os

import pathlib
import pickle
import tempfile

import stable_baselines3 as sb3

from imitation.algorithms import adversarial, bc
from imitation.data import rollout
from imitation.util import logger, util

import pickle
from imitation.data.types import Trajectory
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random

import gym

import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControlDyn import DSLPIDControlDyn
from gym_pybullet_drones.envs.imitation.DynAviaryWGoal import DynAviaryWGoal
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 1)', metavar='')
    parser.add_argument('--physics',            default="dyn",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=False,      type=str2bool,      help='Whether to use PyBullet GUI (default: False)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=False,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=30,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--ctrl_mode',          default="dyn",      type=str)
    ARGS = parser.parse_args()

    dirname = os.path.dirname(__file__)

    num_trajs = 1000

    filename = os.path.join(dirname, "expert_trajectories_vertical_" + str(num_trajs) + ".pkl")
    file = open(filename, "rb")
    trajectories = pickle.load(file)
    transitions = flatten_trajectories(trajectories)

    #### Initialize the simulation #############################
    INIT_XYZS = np.array([[0, 0, 0.02] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0, 0] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Initialize wapoint params ######################
    THRESH_WP = 0.3

    venv = util.make_vec_env("dyn-aviary-w-goal-v0", n_envs=12,
                                wp_thresh=THRESH_WP,
                                drone_model=ARGS.drone,
                                initial_xyzs=INIT_XYZS,
                                initial_rpys=INIT_RPYS,
                                physics=ARGS.physics,
                                neighbourhood_radius=10,
                                freq=ARGS.simulation_freq_hz,
                                aggregate_phy_steps=AGGR_PHY_STEPS,
                                gui=ARGS.gui,
                                record=ARGS.record_video,
                                obstacles=ARGS.obstacles,
                                user_debug_gui=ARGS.user_debug_gui)


    # Train GAIL on expert data.
    # GAIL, and AIRL also accept as `expert_data` any Pytorch-style DataLoader that
    # iterates over dictionaries containing observations, actions, and next_observations.
    gail_logger = logger.configure(os.path.join(dirname,"./GAIL_A2C/" + str(num_trajs)))
    gail_trainer = adversarial.GAIL(
        venv,
        expert_data=transitions,
        expert_batch_size=32,
        gen_algo=sb3.A2C("MlpPolicy", venv, verbose=1),
        custom_logger=gail_logger,
        allow_variable_horizon=True
    )

    n_epochs = 24576
    for i in range(n_epochs, 10000000000000, n_epochs):
        gail_logger = logger.configure(os.path.join(dirname,"./GAIL_A2C/" + str(num_trajs) + "/" + str(i)))
        gail_trainer._logger = gail_logger
        gail_trainer.train(total_timesteps=n_epochs)
        gail_trainer.gen_algo.save(os.path.join(dirname, "./GAIL_A2C/" + str(num_trajs) + "/gail_model_" + str(i) + "_on_" + str(num_trajs) + "_trajs.zip"))
