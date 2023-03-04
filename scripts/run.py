import argparse
import os
import time
import tqdm

import numpy as np
import torch
import gym

from utils import torch_utils as ptu
from utils import utils
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer

from scripts.script_utils import make_logger, make_config

# Import all agents
from agents.dqn_agent import DQNAgent
from agents.td3_agent import TD3Agent
from agents.ppo_agent import PPOAgent

AGENTS = {
    "dqn": DQNAgent,
    "td3": TD3Agent,
    "ppo": PPOAgent,
}

def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.gpu_init(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)


    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)
    ep_len = env.spec.max_episode_steps

    agent_class = AGENTS[args.algo]
    agent = agent_class(
        env.observation_space.shape,
        env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0],
        **config["agent_kwargs"],
    )

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    observation = env.reset()

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        if args.algo in ["td3", "ppo"] and step < config.get("random_steps", 0):
            action = env.action_space.sample()
        else:
            action = agent.get_action(observation)

        next_observation, reward, done, info = env.step(action)
        next_observation = np.asarray(next_observation)
        truncated = info.get("TimeLimit.truncated", False)

        replay_buffer.insert(observation=observation, action=action, reward=reward, next_observation=next_observation, done=done)

        if done:
            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
            observation = env.reset()
        else:
            observation = next_observation

        if step >= config["learning_starts"]:
            batch = replay_buffer.sample(batch_size=config["batch_size"])
            batch = ptu.from_numpy(batch)
            update_info = agent.update(
                obs=batch["observations"],
                action=batch["actions"],
                reward=batch["rewards"],
                next_obs=batch["next_observations"],
                done=batch["dones"],
                step=step,
            )
            if step % args.log_interval == 0:
                update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]
                for k, v in update_info.items():
                    if v is not None:  # Avoid logging None values
                        logger.log_scalar(v, k, step)
                logger.flush()

        if step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(eval_env, agent, args.num_eval_trajectories, ep_len)
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(render_env, agent, args.num_render_trajectories, ep_len, render=True)
                logger.log_paths_as_videos(video_trajectories, step, fps=env.metadata.get("render_fps", 4), max_videos_to_save=args.num_render_trajectories, video_title="eval_rollouts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", "-a", type=str, choices=AGENTS.keys(), required=True, help="Algorithm to run (dqn, td3, ppo)")
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)
    
    args = parser.parse_args()

    logdir_prefix = ""  

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
