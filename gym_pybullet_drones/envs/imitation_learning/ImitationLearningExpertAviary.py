from gym_pybullet_drones.envs.DynAviary import DynAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ObservationType
import numpy as np
from gym import spaces

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics

class ImitationLearningExpertAviary(DynAviary):
    """Class for an expert model used for single agent imitaion learning"""

    def __init__(self):
        """Initialization of an expert model used for single agent imitaion learning.

        Currently only supports torques actions and the task is to fly thru a gate of known location

        Parameters
        ----------

        """

        self.gate_pose = np.array([0, -2, 0.75])

        super().__init__(gui=True)

    ################################################################################
    
    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of key "state", see the implementation of `_getDroneStateVector()`,
        the value of key "neighbors" is the drone's own row of the adjacency matrix,
        the value of key "gate" is the pose of the target gate to fly through.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), Box(3,), MultiBinary(NUM_DRONES)}.

        """
        adjacency_mat = self._getAdjacencyMatrix()
        return {str(i): {"state": self._getDroneStateVector(i), "neighbors": adjacency_mat[i,:], "gate": self.gate_pose.reshape(3,)} for i in range(self.NUM_DRONES) }

    ################################################################################
    
    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), Box(3,), MultiBinary(NUM_DRONES)}.

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.])
        obs_upper_bound = np.array([np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        gate_lower_bound = np.array([-np.inf, -np.inf, -np.inf])
        gate_upper_bound = np.array([np.inf, np.inf, np.inf])
        return spaces.Dict({str(i): spaces.Dict({"state": spaces.Box(low=obs_lower_bound,
                                                                     high=obs_upper_bound,
                                                                     dtype=np.float32
                                                                     ),
                                                 "gate": spaces.Box( low=gate_lower_bound,
                                                                     high=gate_upper_bound,
                                                                     dtype=np.float32),                                                                     
                                                 "neighbors": spaces.MultiBinary(self.NUM_DRONES)
                                                 }) for i in range(self.NUM_DRONES)})

    ################################################################################

    def predict(self,
                obs
                ):
        """ Method that implements the expert actions based on observations it sees

        Parameters
        ----------
        obs: observation

        Returns
        ----------
        act: action
        """

        