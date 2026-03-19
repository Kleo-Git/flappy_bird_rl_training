import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import yaml
from datetime import datetime, timedelta
import argparse
import itertools
import flappy_bird_gymnasium
import os

from policynet import PolicyNet

device = "cuda" if torch.cuda.is_available() else "cpu"

date_format = "%m-%d %H:%M:%S"

runs_dir = "runs"
os.makedirs(runs_dir, exist_ok=True)

matplotlib.use("Agg")

class ReinforceAgent():

    def __init__(self, hp_set):
        with open("hyperparameters.yml", "r") as f:
            all_hp_sets = yaml.safe_load(f)
            hp = all_hp_sets[hp_set]

        self.hp_set = hp_set

        self.env_id = hp["env_id"]
        self.lr = hp["lr"]
        self.g_discount = hp["g_discount"]
        self.stop_on_reward = hp["stop_on_reward"]
        self.hidden_layer_neurons = hp["hidden_layer_neurons"]
        self.env_make_params = hp.get("env_make_params", {})

        self.optimizer = None

        self.log_file = os.path.join(runs_dir, f"{self.hp_set}.log")
        self.model_file = os.path.join(runs_dir, f"{self.hp_set}.pt")
        self.graph_file = os.path.join(runs_dir, f"{self.hp_set}.png")


    def run(self, is_training = True, render = False):
        if is_training:
            start_time = datetime.now()
            last_graph_update = start_time

            log_message = f"{start_time.strftime(date_format)}: Training starting"
            print(log_message)
            with open(self.log_file, 'w') as file:
                file.write(log_message + '\n')

        env = gym.make(self.env_id, render_mode="human" if render else None, **self.env_make_params) # ** unpackages lists

        num_actions = env.action_space.n

        num_states = env.observation_space.shape[0]

        rewards_per_ep = []

        policy_network = PolicyNet(num_states, num_actions, self.hidden_layer_neurons).to(device)

        if is_training:
            self.optimizer = torch.optim.Adam(policy_network.parameters(), lr=self.lr)
            best_reward = -np.inf
        else:
            policy_network.load_state_dict(torch.load(self.model_file))
            policy_network.eval()

        #Allows infinite training
        for episode in itertools.count():
            state, i = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

            if episode % 100 == 0:
                avg_reward = np.mean(rewards_per_ep[-100:]) if len(rewards_per_ep) >= 100 else (np.mean(rewards_per_ep) if rewards_per_ep else 0) 
                print(f"Currently on {episode}, Avg reward = {avg_reward:.1f}") 

            terminated = False
            truncated = False
            episode_r = 0

            log_probs = []
            rewards = []

            while not (terminated or truncated):
                if is_training:
                    action, log_prob = policy_network.act(state.unsqueeze(dim=0))
                    action = action.item()
                    log_probs.append(log_prob) #∇θ ln π(At|St,θt) equation (3)
                else:
                    with torch.no_grad():
                        action_probabilites = policy_network(state.unsqueeze(dim=0))
                        action = action_probabilites.argmax().item()
                
                new_state, reward, terminated, truncated, info = env.step(action)

                episode_r += reward
                
                if is_training:
                    rewards.append(reward)

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                state = new_state
            rewards_per_ep.append(episode_r)

            if is_training:
                if episode_r > best_reward:
                    log_message = f"{datetime.now().strftime(date_format)}: New best reward {episode_r:0.1f} ({(episode_r-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.log_file, "a") as f:
                        f.write(log_message + "\n")
                    
                    torch.save(policy_network.state_dict(), self.model_file)
                    best_reward = episode_r

                current_time = datetime.now()
                if current_time - last_graph_update> timedelta(seconds=10):
                    self.save_graph(rewards_per_ep)
                    last_graph_update = current_time
                
                self.optimize(log_probs, rewards)

    def save_graph(self, rewards_per_episode):
        fig = plt.figure(1)

        mean_r = np.zeros(len(rewards_per_episode))
        for i in range(len(mean_r)):
            mean_r[i] = np.mean(rewards_per_episode[max(0, i-99):(i+1)])
        
        plt.xlabel("Num Episodes")
        plt.ylabel("Mean Rewards")
        plt.plot(mean_r)

        fig.savefig(self.graph_file)
        plt.close(fig)

    def optimize(self, log_probs, rewards):
        returns = []
        g_value = 0

        for reward in reversed(rewards):
            g_value = reward + self.g_discount * g_value #Calculate discounted g value
            returns.insert(0, g_value)

        returns = torch.tensor(returns, dtype=torch.float, device=device)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8) #Normalize returns

        policy_loss = []
        for log_prob, g_value in zip(log_probs, returns):
            policy_loss.append(-log_prob * g_value) #Calculate policy loss

        policy_loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward() #Back prop to learn new weights from policy loss
        self.optimizer.step()



if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    agent = ReinforceAgent(hp_set=args.hyperparameters)

    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(is_training=False, render=True)
    