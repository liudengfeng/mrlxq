import pytest
from muzero.xq_env import XiangQiV0, XiangQiV1
from ray.rllib.env.env_context import EnvContext
import numpy as np
from muzero.game import Game


def test_valid_actions():
    config = EnvContext({}, 0)
    env = XiangQiV0(config)
    obs = {}
    env._fix_action_mask(obs)
    g = Game("", False)
    actions = g.legal_actions_history[-1]
    np.testing.assert_equal(np.sum(env.valid_actions), len(actions))
