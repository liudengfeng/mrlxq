import time

import numpy as np
import pytest
from cppxq import m2a
from ray.rllib.env.env_context import EnvContext

from muzero.constants import *
from muzero.game import Game
from muzero.xq_env import XiangQiV0, XiangQiV1


def test_valid_actions():
    config = EnvContext({}, 0)
    env = XiangQiV0(config)
    obs = {}
    env._fix_action_mask(obs)
    g = Game("", False)
    actions = g.legal_actions_history[-1]
    np.testing.assert_equal(np.sum(env.valid_actions), len(actions))


def test_step_v0():
    env_dict = {"gen_qp": True}
    config = EnvContext(env_dict, 0)
    env = XiangQiV0(config)
    env.reset()
    # env.render("human")
    move = "1242"
    next_state, reward, terminated, truncated, info = env.step(m2a(move))
    assert next_state["obs"].shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
    assert next_state["action_mask"].shape == (NUM_ACTIONS,)
    # env.render("human")
    # time.sleep(3)
