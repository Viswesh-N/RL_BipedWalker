import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical         # contains probability distributions; categorical distribution is used for discrete actions
from torch.optim import Adam                        # Adam optimizer
import gym
from gym.spaces import Discrete, Box            # Discrete is used for discrete actions/observations; Box is used for continuous actions/observations


# Building a feed forward neural network

def mlp(sizes, activation = nn.Tanh, output_activation = nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else output_activation             # if i is not the last layer, use activation; else use output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]             # add linear layer and activation function
    return nn.Sequential(*layers)

# Alternate implementation; in this case, the weighting given during the update step has the rewards only from the time
# instant itself and not the whole trajectory

def reward_to_go(rews):
    n = len(rews)
    rtgs =  np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)               # the rtgs would be a list that contains the rewards from the time instant itself to the end of the episode
    return rtgs                                                         # has been implemented as iterating in the reverse order and appending the from the immediate next timestep
    

# Creating the main training loop

def train(env_name = 'CartPole-v0', hidden_sizes = [32,16], lr = 1e-2, epochs = 50, batch_size= 5000, render = True):

    # creating the environment
    env = gym.make(env_name)

    # checking if the environment is compatible with the algorithm
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."
    
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # creating the policy network
    network = mlp([obs_dim] + hidden_sizes + [n_actions])

    # generating the policy from the network
    def get_policy(obs):
        logits  = network(obs)                      # pass observation through the network to get output logits
        return Categorical(logits=logits)           # pass logits through the categorical distribution to get the policy

    # generating a particular action from the stochastic policy
    def get_action(obs):
        return get_policy(obs).sample().item()     # sample an action from the policy

    # Loss function
    # logp = log probability of the action
    # get_policy(obs) returns the policy(categorical distribution); log_prob(act) returns the log probability of the action
    # the loss is computed as the negative of the weighted average of the log probabilities of the actions

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)       # get_policy(obs) returns the policy(categorical distribution); log_prob(act) returns the log probability of the action
        loss = -(logp * weights).mean()             # computing the loss, where weights is the R(tau) from the policy gradient update step

        return loss

    # creating the optimizer

    optimizer = Adam(network.parameters(), lr = lr)


    def train_one_epoch():
        # collecting trajectories
        batch_obs = []      # for observations
        batch_acts = []     # for actions
        batch_weights = []  # for R(tau) weighting in policy gradient step
        batch_rets = []     # for episode returns
        batch_lens = []     # for episode lengths

        # reseting the environment
        obs = env.reset()
        done = False
        ep_rews = []        # for storing rewards obtained in one episode

        #render the first episode of each epoch
        finised_render_this_epoch = False

        # collecting trajectories; runs for multiple episodes
        while True:

            # render first epoch alone; the finished_render_this_epoch variable is set to True after this
            if(not finised_render_this_epoch and render):
                env.render()
            
            batch_obs.append(obs.copy())        # append the observation to the batch_obs list

            act = get_action(torch.as_tensor(obs, dtype=torch.float32))               # get the action from the policy
            obs, rew, done, _ = env.step(act)   # take the action in the environment

            # save action and reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:

                ep_ret = sum(ep_rews)           # compute return of the episode
                ep_len = len(ep_rews)           # compute length of the episode
                batch_rets.append(ep_ret)        # append the return to the batch_ret list
                batch_lens.append(ep_len)        # append the length to the batch_len list

                # weight for each logprob(a|s) is R(tau)

                # batch_weights += [ep_ret] * ep_len          # creating a list of length ep_len with all elements as ep_ret

                # Uncomment the above line and comment the below line to get the alternate implementation of the algorithm (a more simple one)

                batch_weights += list(reward_to_go(ep_rews))  # Creating a list where in each pass we add the reward to go for the episode

                # reset episode-specific variables
                obs, done, ep_rews, = env.reset(), False, []

                # won't render again this epoch
                finised_render_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:                         # A collection of episodes is a batch; when we have seen over batch_size number of observations, we break the loop
                    break
        # take a single policy gradient update step
        batch_loss = compute_loss(torch.as_tensor(batch_obs, dtype = torch.float32), torch.as_tensor(batch_acts, dtype = torch.int32), torch.as_tensor(batch_weights, dtype = torch.float32))
        batch_loss.backward()
        optimizer.step()
        # print("Action:",act)
        # print("Observation:",obs)
        
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: ', i, 'loss: ', batch_loss.item(), 'return: ', np.mean(batch_rets), 'ep_len: ', np.mean(batch_lens))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type = str, default = 'CartPole-v0')
    parser.add_argument('--render', action = 'store_true')
    parser.add_argument('--lr', type = float, default = 1e-2)
    args = parser.parse_args()
    print("Implementation of Vanilla Policy Gradient\n")
    train(env_name=args.env, lr = args.lr, render = args.render)