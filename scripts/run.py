import os
import time

from typing import Optional 
from matplotlib import pyplot as plt
import yaml

import gym
import numpy as np
import torch

from utils import torch_utils as tu
from utils import utils
import tqdm
import argparse

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--agent", type=str, required=True, default="dqn")
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_traj", "-neval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id" type=int, default=0)
    parser.add_argument("--log_interval",type=int, default=1000)

    args = parser.parse_args()

    logdir_pref = "agent_{}".format(args.agent)

    config = None

