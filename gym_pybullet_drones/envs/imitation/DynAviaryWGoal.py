import numpy as np
from gym import spaces

from gym_pybullet_drones.envs.DynAviary import DroneModel, Physics, DynAviary
from gym_pybullet_drones.utils.utils import nnlsRPM

class DynAviaryWGoal(DynAviary):
    """Multi-drone environment class for control with desired thrust and torques."""

    ################################################################################
    
    def __init__(self,
                 goal_pose,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
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

        self.goal_pose = goal_pose
        super().__init__(drone_model=drone_model,
                 num_drones=num_drones,
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

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of key "state", see the implementation of `_getDroneStateVector()`,
        the value of key "neighbors" is the drone's own row of the adjacency matrix.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES)}.

        """
        adjacency_mat = self._getAdjacencyMatrix()
        return {str(i): {"state": self._getDroneStateVector(i), "goal": self.goal_pose} for i in range(self.NUM_DRONES) }
