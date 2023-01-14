import numpy as np
import gym
import torch
from torch.nn import functional as F
from torch import nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt

plot_result = []


def mlp(sizes, activation=nn.ReLU, end_activation=nn.Identity):
    layers = []
    for i in range (len(sizes)-1):
        act = activation if i < (len(sizes) - 2) else end_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

def rewards_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


class Agent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.episodes = 1500

        self.gamma = 0.95
        # self.epsilon = 1
        # self.epsilon_min = 0.001
        # self.epsilon_decay = 0.9
        self.batch_size = 64
        self.epochs = 750
        self.lr = 0.001

        self.model = mlp([self.state_size, 24, 24, self.action_size])
        self.model.train()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)


    def compute_loss(self, obs, acts, weights):
        # print(weights)
        log_probs = self.get_policy(obs).log_prob(acts)
        return -(log_probs * weights).mean()

    
    def get_policy(self, obs):
        logits = self.model(obs)
        return Categorical(logits=logits)
    

    def act(self,state):
        return self.get_policy(state).sample().item()
        # else:
        #     act_values = self.get_policy(state)
        #     return np.argmax(act_values)

    def plot_results(self, batch_rets):
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.plot(range(len(batch_rets)), batch_rets)
        plt.show()

    def train_one_epoch(self, render = True):

        self.batch_obs = []
        self.batch_acts = []
        self.batch_weights = []
        self.batch_rets = []
        self.batch_lens = []

        self.obs = self.env.reset()
        self.done = False
        self.ep_rews = []

        while True:
            if(render):
                self.env.render()
            
            self.batch_obs.append(self.obs.copy())
            self.action = self.act(torch.as_tensor(self.obs,dtype = torch.float32))
            self.obs, self.reward, self.done, _ = self.env.step(self.action)
            self.ep_rews.append(self.reward)
            self.batch_acts.append(self.action)

            if self.done:
                self.ep_ret = sum(self.ep_rews)
                self.batch_rets.append(self.ep_ret)
                self.ep_len = len(self.ep_rews)
                self.batch_lens.append(self.ep_len)
                self.batch_weights += list(rewards_to_go(self.ep_rews))
                
                self.obs , self.done , self.ep_rews = self.env.reset() , False , []
                

                if len(self.batch_obs) > self.batch_size:
                    break

        self.optimizer.zero_grad()
        self.batch_loss = self.compute_loss(torch.as_tensor(self.batch_obs, dtype=torch.float32), 
                                            torch.as_tensor(self.batch_acts, dtype=torch.long), 
                                            torch.as_tensor(self.batch_weights, dtype=torch.float32))
        self.batch_loss.backward()
        # print("Gradient before step: ", list(self.model.parameters())[0].grad)
        self.optimizer.step()
        # print("Gradient after step: ", list(self.model.parameters())[0].grad)




        return self.batch_rets, self.batch_lens

    
    def train(self):
        for i in range(self.epochs):
            self.batch_rets, self.batch_lens = self.train_one_epoch()
            print(f"Epoch: {i}, Avg Reward: {np.mean(self.batch_rets)}, Avg Length: {np.mean(self.batch_lens)}")
            plot_result.append(np.mean(self.batch_rets))


if __name__ == '__main__':
    print("Implementation of Vanilla Policy Gradient\n")
    agent = Agent()
    agent.train()
    agent.env.close()
    agent.plot_results(plot_result)
    




