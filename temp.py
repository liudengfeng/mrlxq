from gymnasium.spaces import Box, Dict, Discrete
import numpy as np

observation_space = Box(
    0,
    255,
    (10, 9, 3),
    dtype=np.uint8,
)

observation_space = Dict(
    {
        "action_mask": Box(0.0, 1.0, shape=(2086,)),
        "observations": observation_space,
    }
)
obs = observation_space.sample()
action_mask = obs["action_mask"]
for i in range(10):
    action_mask[i] = 1.0
action_mask
