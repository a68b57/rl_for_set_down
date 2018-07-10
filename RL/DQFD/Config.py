# -*- coding: utf-8 -*-
import os


class Config:
    # ENV_NAME = "CartPole-v1"
    ENV_NAME = "SetDown-v2"
    GAMMA = 0.99  # discount factor for target Q
    INITIAL_EPSILON = 1.0  # starting value of epsilon
    FINAL_EPSILON = 0.01  # final value of epsilon
    EPSILIN_DECAY = 0.9999
    START_TRAINING = 1000 # num of experiences in buffer before updating (deprecated in V3)
    BATCH_SIZE = 32  # size of minibatch
    UPDATE_TARGET_NET = 10000 # update target every 10000 steps
    LEARNING_RATE = 0.001
    DEMO_RATIO = 0.2
    LAMBDA = [1.0, 0.0, 1.0, 10e-5]  # for [loss_dq, loss_n_dq, loss_jeq, loss_l2]
    PRETRAIN_STEPS = 10000
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/24.2_DQfD_2_model')
    DEMO_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'expert_data/demo.p')
    SAVE_MODEL = 100000 # save model every 100000 steps

    demo_buffer_size = 123994
    replay_buffer_size = demo_buffer_size * 2
    iteration = 1
    episode = 10000
    trajectory_n = 10  # for n-step TD-loss (both demo data and generated data)


class DDQNConfig(Config):
    demo_mode = 'get_demo'


class DQfDConfig(Config):
    demo_mode = 'use_demo'
    demo_num = int(Config.BATCH_SIZE * Config.DEMO_RATIO)


