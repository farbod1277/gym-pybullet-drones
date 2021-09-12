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
from gym_pybullet_drones.envs.DynAviary import DynAviary
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
    parser.add_argument('--obstacles',          default=True,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=12,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--ctrl_mode',          default="dyn",      type=str)
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    INIT_XYZS = np.array([[0, 0, 0] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0, 0] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Initialize wapoint at centre of gate ######################
    PERIOD = 10
    NUM_WP = 2
    wp_counter = 0
    TARGET_POS = np.zeros((NUM_WP,3))
    TARGET_POS[0, :] = np.array([2, 0, 1])
    TARGET_POS[1, :] = np.array([3, 5, 4])

    #### Create the environment ##
    if ARGS.ctrl_mode == "dyn":
        env = DynAviary(drone_model=ARGS.drone,
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
            ctrl = [DSLPIDControlDyn(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]
        else:
            ctrl = [DSLPIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    action = {str(i): np.array([0,0,0,0]) for i in range(ARGS.num_drones)}
    START = time.time()
    for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        if i >= int(5*env.SIM_FREQ): wp_counter = 1

        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            for j in range(ARGS.num_drones):
                action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                       state=obs[str(j)]["state"],
                                                                       target_pos=TARGET_POS[wp_counter, :],
                                                                       target_rpy=INIT_RPYS[j, :]
                                                                       )

        #### Log the simulation ####################################
        for j in range(ARGS.num_drones):
            logger.log(drone=j,
                       timestamp=i/env.SIM_FREQ,
                       state= obs[str(j)]["state"],
                       control=np.hstack([TARGET_POS[0, 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        if i%env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Plot the simulation results ###########################
    if ARGS.plot:
        logger.plot()
