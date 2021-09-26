"""Script used for collecting and storing trajectories from expert control model for imitation learning.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python expert_trajectories.py

Notes
-----
The drone moves toward the centre of a gate randomly generated at different locations.

"""
import pickle
from imitation.data.types import Trajectory
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random


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
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
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

    #### Initialize the simulation #############################
    INIT_XYZS = np.array([[0, 0, 0] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0, 0] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Initialize wapoint params ######################
    THRESH_WP = 0.4

    #### Extract waypoints from csv
    dirname = os.path.dirname(__file__)
    TARGET_POS = np.genfromtxt(os.path.join(dirname, 'paths_circle.csv'), delimiter=',')
    NUM_WP = np.shape(TARGET_POS)[0]

    #### Create the environment ##
    if ARGS.ctrl_mode == "dyn":
        env = DynAviaryWGoal(num_wps=NUM_WP,
                            wp_thresh=THRESH_WP,
                            goal_poses=TARGET_POS,
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
                            user_debug_gui=ARGS.user_debug_gui
                            )
    else:
        env = CtrlAviary(drone_model=ARGS.drone,
                    num_drones=ARGS.num_drones,
                    initial_xyzs=INIT_XYZS,
                    initial_rpys=INIT_RPYS,
                    physics=ARGS.physics,
                    neighbourhood_radius=10,
                    freq=ARGS.simulation_freq_hz,
                    aggregate_phy_steps=AGGR_PHY_STEPS,
                    gui=ARGS.gui,
                    record=ARGS.record_video,
                    obstacles=ARGS.obstacles,
                    user_debug_gui=ARGS.user_debug_gui
                    )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=ARGS.num_drones
                    )

    #### Initialize the controllers ############################
    if ARGS.drone in [DroneModel.CF2X, DroneModel.CF2P]:
        if ARGS.ctrl_mode == "dyn":
            ctrl = DSLPIDControlDyn(drone_model=ARGS.drone)
        else:
            ctrl = DSLPIDControl(drone_model=ARGS.drone)

    #### Initialize data structure for storing expert Trajectories
    trajectories = Trajectory(obs=np.zeros((int(ARGS.duration_sec*env.SIM_FREQ/AGGR_PHY_STEPS),23)),
                              acts=np.zeros((int(ARGS.duration_sec*env.SIM_FREQ/AGGR_PHY_STEPS) - 1,4)),
                              infos=np.zeros((int(ARGS.duration_sec*env.SIM_FREQ/AGGR_PHY_STEPS) - 1,1)),
                              terminal=True)

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    action = np.array([0,0,0,0])
    START = time.time()
    for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
        
        #### Step the simulation ###################################
        obs, _, done, _ = env.step(action)

        trajectories.obs[int(i/AGGR_PHY_STEPS), :] = obs
        if int(i/5) < ((int(ARGS.duration_sec*env.SIM_FREQ/AGGR_PHY_STEPS) - 1)): trajectories.acts[int(i/AGGR_PHY_STEPS), :] = action

        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            action, _, _ = ctrl.computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                    state=obs[0:20],
                                                                    target_pos=obs[20:23],
                                                                    target_rpy=INIT_RPYS[0, :]
                                                                    )

        #### Log the simulation ####################################
        for j in range(ARGS.num_drones):
            logger.log(drone=j,
                       timestamp=i/env.SIM_FREQ,
                       state= obs[0:20],
                       control=np.hstack([TARGET_POS[0, 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        if i%env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

        if done:
            break

    #### Save trajectories in a pickle file ########################
    with open(os.path.join(dirname, 'expert_trajectories.pkl'), "wb") as f:
        pickle.dump([trajectories], f)

    #### Close the environment #################################
    env.close()

    #### Plot the simulation results ###########################
    if ARGS.plot:
        logger.plot()
