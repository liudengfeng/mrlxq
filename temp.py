from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
from muzero.xq_env import XiangQiV0

config = AlphaZeroConfig()
config = config.training(sgd_minibatch_size=256)
config = config.resources(num_gpus=0)
config = config.rollouts(num_rollout_workers=2)
config = config.environment(
    XiangQiV0,
    env_config={
        "gen_qp": True,
    },
)
print(config.to_dict())
# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build()
algo.train()
