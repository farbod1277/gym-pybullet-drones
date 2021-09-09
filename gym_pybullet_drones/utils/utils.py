"""General use functions.
"""
import time
import argparse
from typing import Sequence, Union

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.envs.DynAviary import DynAviary

from imitation.data import types
from imitation.data.rollout import GenTrajTerminationFn, TrajectoryAccumulator

import numpy as np
from scipy.optimize import nnls

from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.vec_env import VecEnv

################################################################################

def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > .04 or i%(int(1/(24*timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i*timestep):
            time.sleep(timestep*i - elapsed)

################################################################################

def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")

################################################################################

def nnlsRPM(thrust,
            x_torque,
            y_torque,
            z_torque,
            counter,
            max_thrust,
            max_xy_torque,
            max_z_torque,
            a,
            inv_a,
            b_coeff,
            gui=False
            ):
    """Non-negative Least Squares (NNLS) RPMs from desired thrust and torques.

    This function uses the NNLS implementation in `scipy.optimize`.

    Parameters
    ----------
    thrust : float
        Desired thrust along the drone's z-axis.
    x_torque : float
        Desired drone's x-axis torque.
    y_torque : float
        Desired drone's y-axis torque.
    z_torque : float
        Desired drone's z-axis torque.
    counter : int
        Simulation or control iteration, only used for printouts.
    max_thrust : float
        Maximum thrust of the quadcopter.
    max_xy_torque : float
        Maximum torque around the x and y axes of the quadcopter.
    max_z_torque : float
        Maximum torque around the z axis of the quadcopter.
    a : ndarray
        (4, 4)-shaped array of floats containing the motors configuration.
    inv_a : ndarray
        (4, 4)-shaped array of floats, inverse of a.
    b_coeff : ndarray
        (4,1)-shaped array of floats containing the coefficients to re-scale thrust and torques. 
    gui : boolean, optional
        Whether a GUI is active or not, only used for printouts.

    Returns
    -------
    ndarray
        (4,)-shaped array of ints containing the desired RPMs of each propeller.

    """
    #### Check the feasibility of thrust and torques ###########
    if gui and thrust < 0 or thrust > max_thrust:
        print("[WARNING] iter", counter, "in utils.nnlsRPM(), unfeasible thrust {:.2f} outside range [0, {:.2f}]".format(thrust, max_thrust))
    if gui and np.abs(x_torque) > max_xy_torque:
        print("[WARNING] iter", counter, "in utils.nnlsRPM(), unfeasible roll torque {:.2f} outside range [{:.2f}, {:.2f}]".format(x_torque, -max_xy_torque, max_xy_torque))
    if gui and np.abs(y_torque) > max_xy_torque:
        print("[WARNING] iter", counter, "in utils.nnlsRPM(), unfeasible pitch torque {:.2f} outside range [{:.2f}, {:.2f}]".format(y_torque, -max_xy_torque, max_xy_torque))
    if gui and np.abs(z_torque) > max_z_torque:
        print("[WARNING] iter", counter, "in utils.nnlsRPM(), unfeasible yaw torque {:.2f} outside range [{:.2f}, {:.2f}]".format(z_torque, -max_z_torque, max_z_torque))
    B = np.multiply(np.array([thrust, x_torque, y_torque, z_torque]), b_coeff)
    sq_rpm = np.dot(inv_a, B)
    #### NNLS if any of the desired ang vel is negative ########
    if np.min(sq_rpm) < 0:
        sol, res = nnls(a,
                        B,
                        maxiter=3*a.shape[1]
                        )
        if gui:
            print("[WARNING] iter", counter, "in utils.nnlsRPM(), unfeasible squared rotor speeds, using NNLS")
            print("Negative sq. rotor speeds:\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(sq_rpm[0], sq_rpm[1], sq_rpm[2], sq_rpm[3]),
                   "\t\tNormalized: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(sq_rpm[0]/np.linalg.norm(sq_rpm), sq_rpm[1]/np.linalg.norm(sq_rpm), sq_rpm[2]/np.linalg.norm(sq_rpm), sq_rpm[3]/np.linalg.norm(sq_rpm)))
            print("NNLS:\t\t\t\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(sol[0], sol[1], sol[2], sol[3]),
                  "\t\t\tNormalized: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(sol[0]/np.linalg.norm(sol), sol[1]/np.linalg.norm(sol), sol[2]/np.linalg.norm(sol), sol[3]/np.linalg.norm(sol)),
                  "\t\tResidual: {:.2f}".format(res))
        sq_rpm = sol
    return np.sqrt(sq_rpm)


def generate_trajectories(
    expert,
    venv: VecEnv,
    sample_until: GenTrajTerminationFn,
    *,
    deterministic_policy: bool = False,
    rng: np.random.RandomState = np.random,
) -> Sequence[types.TrajectoryWithRew]:
    """Generate trajectory dictionaries from a policy and an environment.

    Args:
      expert (BaseAviary): A BaseAviary subclass implementing expert model
      venv: The vectorized environments to interact with.
      sample_until: A function determining the termination condition.
          It takes a sequence of trajectories, and returns a bool.
          Most users will want to use one of `min_episodes` or `min_timesteps`.
      deterministic_policy: If True, asks policy to deterministically return
          action. Note the trajectories might still be non-deterministic if the
          environment has non-determinism!
      rng: used for shuffling trajectories.

    Returns:
      Sequence of trajectories, satisfying `sample_until`. Additional trajectories
      may be collected to avoid biasing process towards short episodes; the user
      should truncate if required.
    """
    get_action = expert.predict
    if isinstance(expert, DynAviary):
        # check that the observation and action spaces of policy and environment match
        check_for_correct_spaces(venv, expert.observation_space, expert.action_space)

    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = TrajectoryAccumulator()
    obs = venv.reset()
    for env_idx, ob in enumerate(obs):
        # Seed with first obs only. Inside loop, we'll only add second obs from
        # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
        # get all observations, but they're not duplicated into "next obs" and
        # "previous obs" (this matters for, e.g., Atari, where observations are
        # really big).
        trajectories_accum.add_step(dict(obs=ob), env_idx)

    # Now, we sample until `sample_until(trajectories)` is true.
    # If we just stopped then this would introduce a bias towards shorter episodes,
    # since longer episodes are more likely to still be active, i.e. in the process
    # of being sampled from. To avoid this, we continue sampling until all epsiodes
    # are complete.
    #
    # To start with, all environments are active.
    active = np.ones(venv.num_envs, dtype=bool)
    while np.any(active):
        acts, _ = get_action(obs)
        obs, rews, dones, infos = venv.step(acts)

        # If an environment is inactive, i.e. the episode completed for that
        # environment after `sample_until(trajectories)` was true, then we do
        # *not* want to add any subsequent trajectories from it. We avoid this
        # by just making it never done.
        dones &= active

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            acts, obs, rews, dones, infos
        )
        trajectories.extend(new_trajs)

        if sample_until(trajectories):
            # Termination condition has been reached. Mark as inactive any environments
            # where a trajectory was completed this timestep.
            active &= ~dones

    # Note that we just drop partial trajectories. This is not ideal for some
    # algos; e.g. BC can probably benefit from partial trajectories, too.

    # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
    # `trajectories` sooner. Shuffle to avoid bias in order. This is important
    # when callees end up truncating the number of trajectories or transitions.
    # It is also cheap, since we're just shuffling pointers.
    rng.shuffle(trajectories)

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        exp_obs = (n_steps + 1,) + venv.observation_space.shape
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    return trajectories


def rollout_and_save(
    path: str,
    expert: DynAviary,
    venv: VecEnv,
    sample_until: GenTrajTerminationFn,
    *,
    unwrap: bool = True,
    exclude_infos: bool = True,
    verbose: bool = True,
    **kwargs,
) -> None:
    """Generate expert rollouts and save them to a pickled list of trajectories.

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
      path: Rollouts are saved to this path.
      venv: The vectorized environments.
      sample_until: End condition for rollout sampling.
      unwrap: If True, then save original observations and rewards (instead of
        potentially wrapped observations and rewards) by calling
        `unwrap_traj()`.
      exclude_infos: If True, then exclude `infos` from pickle by setting
        this field to None. Excluding `infos` can save a lot of space during
        pickles.
      verbose: If True, then print out rollout stats before saving.
      deterministic_policy: Argument from `generate_trajectories`.
    """
    trajs = generate_trajectories(expert, venv, sample_until, **kwargs)

    types.save(path, trajs)
