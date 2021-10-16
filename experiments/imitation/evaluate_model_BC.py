from imitation.algorithms import bc
import os
import csv

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
import stable_baselines3 as sb3

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
from stable_baselines3.common.policies import BaseModel
from torch._C import import_ir_module

import signal

def handler(signum, frame):
    results.close()
    exit(1)


signal.signal(signal.SIGINT, handler)

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 1)', metavar='')
    parser.add_argument('--physics',            default="dyn",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=False,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=False,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=False,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=30,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--ctrl_mode',          default="dyn",      type=str)
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    INIT_XYZS = np.array([[0, 0, 0.02] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0, 0] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Initialize wapoint params ######################
    THRESH_WP = 0.3
    init_step = 100
    max_model = 100000

    #### Extract waypoints from csv
    dirname = os.path.dirname(__file__)
    results = open(os.path.join(dirname, "BC/1000/results_" + str(init_step) + "_" + str(max_model) + ".csv"), 'w')
    writer = csv.writer(results, delimiter=",")
    
    for model_step in range(init_step, max_model+1, 100):

        success = 0
        num_eval_steps = 1000

        if model_step > 0:
            try:
                model = bc.reconstruct_policy(os.path.join(dirname, "BC/1000/bc_model_" + str(model_step) + "_on_1000_trajs.pt"))
            except:
                results.close()
                exit(1)
        
        for eval_step in range(0, num_eval_steps):

            # #### Create the environment ##
            if ARGS.ctrl_mode == "dyn":
                env = gym.make('dyn-aviary-w-goal-v0',
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

            #### Initialize the logger #################################
            # logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
            #                 num_drones=1
            #                 )

            #### Run the simulation ####################################
            CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
            action = np.array([0,0,0,0])
            START = time.time()
            for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
                
                #### Step the simulation ###################################
                obs, _, done, _ = env.step(action)

                #### Compute control at the desired frequency ##############
                if i%CTRL_EVERY_N_STEPS == 0:

                    #### Compute control for the current way point #############
                    if model_step > 0:
                        action, _ = model.predict(observation=obs)
                    else:
                        action = np.random.rand(), (np.random.rand() * 2.) - 1., (np.random.rand() * 2.) - 1., (np.random.rand() * 2.) - 1.

                if done:
                    break

                ### Log the simulation ####################################
                # logger.log(drone=0,
                #             timestamp=i/env.SIM_FREQ,
                #             state=obs[0:20]
                #             )

            env.close()
            done = False

            if(np.linalg.norm(obs[20:23] - obs[0:3]) <= THRESH_WP):
                success += 1

            print("\n\n\n\nModel Step: " + str(model_step) + "\nEvaluation: " + str(eval_step) + "\nSuccess: " + str(success) + "\n\n\n\n\n")

            # print(obs[20:23])
            # logger.plot()

        print("Success Rate (" + str(model_step) + "): " + str(success/num_eval_steps))
        writer.writerow([str(model_step), str(success/num_eval_steps)])

        time.sleep(1000)

    results.close()