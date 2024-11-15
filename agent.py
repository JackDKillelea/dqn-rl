from random import random
from dqn import DQN
from experience_replay import ReplayMemory
from datetime import datetime, timedelta
from torch import nn
import argparse
import gymnasium
import itertools
import yaml
import torch
import os
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import flappy_bird_gymnasium

# Date and time format for logging
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run statistics
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

mp.use("Agg")

# Deep-Q Learning Agent
class Agent:
    def __init__(self, hyperparameter_set):
        with open("hyperparameters.yaml", "r") as file:
            all_hyperparameters = yaml.safe_load(file)
            hyperparameters = all_hyperparameters[hyperparameter_set]
            print(hyperparameters)

        # Hyperparameters
        self.name = hyperparameters["name"]
        self.env_id = hyperparameters["env_id"]                             # Environment ID
        self.replay_memory_size = hyperparameters["replay_memory_size"]     # Size of the replay memory - around 100,000 for smaller games such as Cart Pole
        self.batch_size = hyperparameters["batch_size"]                     # Size of the training data set from the replay memory
        self.epsilon_init = hyperparameters["epsilon_init"]                 # 1 = 100% random action, 0 = 0% random action
        self.epsilon_decay = hyperparameters["epsilon_decay"]               # Epsilon decay rate
        self.epsilon_min = hyperparameters["epsilon_min"]                   # Minimum epsilon value
        self.network_sync_rate = hyperparameters["network_sync_rate"]       # Rate at which the target network is updated
        self.discount_factor_q = hyperparameters["discount_factor_q"]       # Discount factor for Q-learning
        self.learning_rate_actor = hyperparameters["learning_rate_actor"]   # Learning rate for actor network
        self.stop_on_reward = hyperparameters["stop_on_reward"]             # Break point if AI gets a reward above the stop_on_reward threshold
        self.fc1_nodes = hyperparameters["fc1_nodes"]                       # Number of nodes to pass for the DQN
        self.enable_dueling_dqn = hyperparameters["enable_dueling_dqn"]     # Enable dueling DQN

        self.loss_function = nn.MSELoss()                                   #  MSE = Mean Squared Error
        self.optimiser = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.name}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.name}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{self.name}.png")

    def run(self, is_training=False, render=False):
        env = gymnasium.make(self.env_id, render_mode="human" if render else None)

        # Get observation space size
        number_states = env.observation_space.shape[0]
        # Get number of possible actions
        number_actions = env.action_space.n

        # Initialize tracking variables
        reward_per_episode = []
        epsilon_history = []
        last_graph_update_time = datetime.now()

        # Set up DQN model
        policy_dqn = DQN(number_states, number_actions, self.fc1_nodes, self.enable_dueling_dqn)

        if is_training:
            # If training, train the target DQN
            replay_memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(number_states, number_actions, self.fc1_nodes, self.enable_dueling_dqn)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_counter = 0
            best_reward = -999999

            self.optimiser = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_actor)
        else:
            # If not training, load the model from file location
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        # Run agent indefinitely
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                if is_training and random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                episode_reward += reward
                new_state = torch.tensor(new_state, dtype=torch.float32)
                reward = torch.tensor(reward, dtype=torch.float32)

                if is_training:
                    replay_memory.append((state, action, new_state, reward, terminated))
                    step_counter += 1

                state = new_state

            reward_per_episode.append(episode_reward)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            if is_training:
                current_time = datetime.now()
                if episode_reward > best_reward:
                    log_message = f"New best reward: Current date: {current_time.strftime(DATE_FORMAT)} Episode: {episode}, Reward: {episode_reward:.2f} , Epsilon: {epsilon:.2f}"
                    print(log_message)
                    with open(self.LOG_FILE, "a") as file:
                        file.write(log_message + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                if current_time - last_graph_update_time > timedelta(seconds=60):
                    self.save_graph(reward_per_episode, epsilon_history)
                    last_graph_update_time = current_time

            # If enough experiences have been collected
            if len(replay_memory) >= self.batch_size:
                # Sample a batch of experiences from the replay memory
                batch = replay_memory.sample(self.batch_size)

                self.optimise(batch, policy_dqn, target_dqn)

                # Copy the weights of the target network to the policy network
                if step_counter > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_counter = 0

    def optimise(self, batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminated = zip(*batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminated = torch.tensor(terminated).float()

        with torch.no_grad():
            # Calculate the target Q-values for the new states
            target_q = rewards + (1 - terminated) * self.discount_factor_q * target_dqn(new_states).max(dim=1)[0]

        # Calculate Q value from current policy
        current_q = policy_dqn(states).gather(1, actions.unsqueeze(dim=1)).squeeze()

        # Compute loss for the current batch
        loss = self.loss_function(current_q, target_q)

        # Clear gradients
        self.optimiser.zero_grad()
        # Compute gradients
        loss.backward()
        # Update network parameters - weights and biases
        self.optimiser.step()

    def save_graph(self, reward_per_episode, epsilon_history):
        fig = plt.figure(figsize=(16, 6))

        mean_reward = np.zeros(len(reward_per_episode))
        for x in range(len(reward_per_episode)):
            mean_reward[x] = np.mean(reward_per_episode[max(0, x-99):x+1])
        plt.subplot(121)
        plt.ylabel("Mean Reward")
        plt.xlabel("Episode")
        plt.plot(mean_reward)

        plt.subplot(122)
        plt.ylabel("Epsilon Decay")
        plt.xlabel("Episode")
        plt.plot(epsilon_history)

        plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9, wspace=0.3, hspace=0.2)

        fig.savefig(self.GRAPH_FILE)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the DQN agent")
    parser.add_argument("hyperparameter", help="")
    parser.add_argument("--train", action="store_true", help="Training Mode")
    args = parser.parse_args()

    dlq = Agent(hyperparameter_set=args.hyperparameter)

    if args.train:
        dlq.run(is_training=True, render=True)
    else:
        dlq.run(is_training=False, render=True)

    # Uncomment the following lines to test specific game
    # agent = Agent("ALE/Breakout-v5")
    # agent.run(is_training=True, render=True)