from gym.envs.registration import register


register(
    id='SetDown-v0',
    entry_point='RL.envs:SetDown',
)


register(
    id='SetDown-v1',
    entry_point='RL.envs:SetDown_gym',
)

register(
    id='Following-v0',
    entry_point='RL.envs:Following',
)


register(
    id='Following-v1',
    entry_point='RL.envs:Following_gym',
)


register(
    id='HRL-v0',
    entry_point='RL.envs:HRL_gym',
)
