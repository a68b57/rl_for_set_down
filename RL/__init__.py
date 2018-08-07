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


register(
    id='SetDown-v2',
    entry_point='RL.envs:SetDown_reimpact_gym',
)

register(
    id='SetDown-v3',
    entry_point='RL.envs:SetDown_2d_gym',
)

register(
    id='SetDown-v4',
    entry_point='RL.envs:SetDown_hmc_insite',
)


register(
	id='SetDown-v5',
	entry_point='RL.envs:SetDown_reimpact_one_step',
)
