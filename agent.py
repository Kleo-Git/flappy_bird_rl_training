import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
import yaml
import os
import itertools
from datetime import datetime, timedelta
 
# Project specific imports
from experience_replay import ReplayMemory
from dqn import DQN
import flappy_bird_gymnasium
 
# Use 'Agg' backend to save images to disk without a GUI window
matplotlib.use("Agg")
 
class Agent:
    def __init__(self, hyperparameter_set):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hp_set = hyperparameter_set
        # Load hyperparameters from YML
        self.config = self._load_config(hyperparameter_set)
 
        # 1. SETUP LOGGING DIRECTORIES AND FILES
        self.runs_dir = "runs"
        os.makedirs(self.runs_dir, exist_ok=True)
        self.log_file = os.path.join(self.runs_dir, f"{self.hp_set}.log")   # Text record
        self.model_file = os.path.join(self.runs_dir, f"{self.hp_set}.pt")  # Saved weights
        self.graph_file = os.path.join(self.runs_dir, f"{self.hp_set}.png") # Visual graph
 
        # Auto-detect environment dimensions
        temp_env = gym.make(self.config['env_id'], **self.config.get('env_make_params', {}))
        self.config['state_dim'] = temp_env.observation_space.shape[0]
        self.config['action_dim'] = temp_env.action_space.n
        temp_env.close()
        # Initialize Networks
        self.policy_net = DQN(
            self.config['state_dim'], 
            self.config['action_dim'], 
            self.config['fc1_nodes'], 
            self.config['enable_dueling_dqn']
        ).to(self.device)
        self.target_net = DQN(
            self.config['state_dim'], 
            self.config['action_dim'], 
            self.config['fc1_nodes'], 
            self.config['enable_dueling_dqn']
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config['learning_rate_a'])
        self.memory = ReplayMemory(self.config['replay_memory_size'])
        self.loss_fn = nn.MSELoss()
 
    def _load_config(self, set_name):
        with open('hyperparameters.yml', 'r') as f:
            cfg = yaml.safe_load(f)[set_name]
        return cfg
 
    def select_action(self, state, epsilon, env):
        if np.random.random() < epsilon:
            return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)
 
    # 2. PLOTTING METHOD
    def save_graph(self, rewards_per_episode, epsilon_history):
        """Creates the visual training log (PNG) showing progress."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # Plot Mean Rewards (Last 100 episodes)
        mean_r = np.zeros(len(rewards_per_episode))
        for i in range(len(mean_r)):
            mean_r[i] = np.mean(rewards_per_episode[max(0, i-99):(i+1)])
        ax1.plot(mean_r, color='blue')
        ax1.set_title("Mean Rewards (Last 100)")
        ax1.set_xlabel("Episodes")
        # Plot Epsilon Decay
        ax2.plot(epsilon_history, color='orange')
        ax2.set_title("Epsilon Decay")
        ax2.set_xlabel("Steps")
        plt.tight_layout()
        fig.savefig(self.graph_file) # Save visual log to disk
        plt.close(fig)
 
    def optimize(self):
        if len(self.memory) < self.config['mini_batch_size']:
            return
 
        transitions = self.memory.sample(self.config['mini_batch_size'])
        batch = list(zip(*transitions))
 
        state_b = torch.cat(batch[0])
        action_b = torch.cat(batch[1])
        new_state_b = torch.cat(batch[2])
        reward_b = torch.cat(batch[3])
        done_b = torch.tensor(batch[4], dtype=torch.float, device=self.device)
 
        current_q = self.policy_net(state_b).gather(1, action_b)
 
        with torch.no_grad():
            if self.config['enable_double_dqn']:
                best_actions = self.policy_net(new_state_b).max(1)[1].unsqueeze(1)
                next_q = self.target_net(new_state_b).gather(1, best_actions).squeeze()
            else:
                next_q = self.target_net(new_state_b).max(1)[0]
            target_q = reward_b + (1 - done_b) * self.config['discount_factor_g'] * next_q
 
        loss = self.loss_fn(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
 
    def train(self):
        env = gym.make(self.config['env_id'], **self.config.get('env_make_params', {}))
        epsilon = self.config['epsilon_init']
        step_count = 0
        best_reward = -np.inf
        # Buffers for visual logging
        rewards_per_episode = []
        epsilon_history = []
        # Timer for graph updates
        last_graph_update = datetime.now()
        # 3. STARTING TEXT LOG
        start_msg = f"{datetime.now().strftime('%m-%d %H:%M:%S')}: Training starting..."
        print(start_msg)
        with open(self.log_file, 'w') as f:
            f.write(start_msg + '\n')
 
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            episode_reward = 0
 
            for t in itertools.count():
                action = self.select_action(state.unsqueeze(0), epsilon, env)
                new_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                episode_reward += reward
                new_state_t = torch.tensor(new_state, dtype=torch.float, device=self.device)
                reward_t = torch.tensor([reward], dtype=torch.float, device=self.device)
 
                self.memory.append(state.unsqueeze(0), action, new_state_t.unsqueeze(0), reward_t, done)
                state = new_state_t
 
                self.optimize()
                epsilon = max(self.config['epsilon_min'], epsilon * self.config['epsilon_decay'])
                epsilon_history.append(epsilon)
                step_count += 1
                if step_count % self.config['network_sync_rate'] == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
 
                if done:
                    break
            rewards_per_episode.append(episode_reward)
 
            # 4. MILESTONE LOGGING: RECORD BEST REWARDS AND SAVE MODEL
            if episode_reward > best_reward:
                best_reward = episode_reward
                log_msg = f"{datetime.now().strftime('%H:%M:%S')}: New best reward {best_reward:.1f} at episode {episode}"
                print(log_msg)
                with open(self.log_file, 'a') as f: # Append to text log
                    f.write(log_msg + '\n')
                # Save the model brain as a log
                torch.save(self.policy_net.state_dict(), self.model_file)
 
            # 5. PERIODIC VISUAL LOG UPDATE (every 10s)
            if datetime.now() - last_graph_update > timedelta(seconds=10):
                self.save_graph(rewards_per_episode, epsilon_history)
                last_graph_update = datetime.now()
            if episode % 50 == 0:
                print(f"Episode {episode} | Reward: {episode_reward:.1f} | Epsilon: {epsilon:.4f}")
 
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agent.py <hyperparameter_set>")
        sys.exit(1)
    agent = Agent(sys.argv[1])
    agent.train()