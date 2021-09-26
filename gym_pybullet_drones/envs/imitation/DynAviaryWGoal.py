import numpy as np
from gym import spaces

from gym_pybullet_drones.envs.DynAviary import DroneModel, Physics, DynAviary
from gym_pybullet_drones.utils.utils import nnlsRPM

class DynAviaryWGoal(DynAviary):
    """Single-drone environment class for control with desired thrust and torques."""

    ################################################################################
    
    def __init__(self,
                 num_wps,
                 wp_thresh,
                 goal_poses,
                 drone_model: DroneModel=DroneModel.CF2X,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 ):

        self.num_wps = num_wps
        self.current_wp = 0
        self.waypoint_thresh = wp_thresh
        self.goal_poses = goal_poses
        super().__init__(drone_model=drone_model,
                 neighbourhood_radius=neighbourhood_radius,
                 initial_xyzs=initial_xyzs,
                 initial_rpys=initial_rpys,
                 physics=physics,
                 freq=freq,
                 aggregate_phy_steps=aggregate_phy_steps,
                 gui=gui,
                 record=record,
                 obstacles=obstacles,
                 user_debug_gui=user_debug_gui,)

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        Box
            A Box(4,)

        """
        #### Action vector ######## Thrust           X Torque             Y Torque             Z Torque
        act_lower_bound = np.array([0.,              -1.,                 -1.,                 -1.])
        act_upper_bound = np.array([1.,              1.,                  1.,                  1.])
        return spaces.Box(low=act_lower_bound,
                          high=act_upper_bound,
                          dtype=np.float32)

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Solves desired thrust and torques using NNLS and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : ndarray
            The input action each drone (desired thrust and torques), to be translated into RPMs.

        Returns
        -------
        ndarray
            (1, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        clipped_action = np.zeros((1, 4))
        clipped_action[0, :] = nnlsRPM(thrust=action[0]*self.MAX_THRUST,
                                            x_torque=action[1]*self.MAX_XY_TORQUE,
                                            y_torque=action[2]*self.MAX_XY_TORQUE,
                                            z_torque=action[3]*self.MAX_Z_TORQUE,
                                            counter=self.step_counter,
                                            max_thrust=self.MAX_THRUST,
                                            max_xy_torque=self.MAX_XY_TORQUE,
                                            max_z_torque=self.MAX_Z_TORQUE,
                                            a=self.A,
                                            inv_a=self.INV_A,
                                            b_coeff=self.B_COEFF,
                                            gui=self.GUI
                                            )
        return clipped_action
    
    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            Box(23,)

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3        goal_pose_x     goal_pose_y     goal_pose_z
        obs_lower_bound = np.array([-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.,       -np.inf,        -np.inf,        -np.inf])
        obs_upper_bound = np.array([np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, np.inf,     np.inf,         np.inf])
        return spaces.Box(low=obs_lower_bound,
                          high=obs_upper_bound,
                          dtype=np.float32)

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of key "state", see the implementation of `_getDroneStateVector()`,
        the value of key "neighbors" is the drone's own row of the adjacency matrix.

        Returns
        -------
        ndarray
            Box(23,)

        """

        drone_state = self._getDroneStateVector(0)
        
        if ((self.current_wp < self.num_wps - 1) &
            (np.linalg.norm(self.goal_poses[self.current_wp] - drone_state[0:3]) <= self.waypoint_thresh)):

            self.current_wp += 1

        return np.concatenate((drone_state, self.goal_poses[self.current_wp]))


    def _computeDone(self):
        """Computes the current done value(s).


        Returns
        -------
        bool

        """
        if(self.current_wp == self.num_wps - 1):
            return True
        else:
            return False
