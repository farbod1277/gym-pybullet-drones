"""Script used for collecting expert trajectories and then training using imitation learning to fly thru a gate.

"""

from imitation.data.rollout import make_sample_until
from imitation.util.util import make_vec_env
from gym_pybullet_drones.envs.imitation_learning.ImitationLearningExpertAviary import ImitationLearningExpertAviary
from gym_pybullet_drones.utils.utils import rollout_and_save
import numpy as np


if __name__ == "__main__":
    NUM_EPISODES = 100
    EPISODE_LEN = 240

    expert = ImitationLearningExpertAviary()
    venv = make_vec_env('imitation-learning-expert-aviary-v0', 1)
    sample_until = make_sample_until(EPISODE_LEN, NUM_EPISODES = 100)
    rollout_and_save("expert/test.pkl", expert, venv, sample_until)