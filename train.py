import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from collections import deque

from blackjack import Deck, InfiniteDeck, EuropeGame, AmericaGame


class FNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.001
        self.batch_size = 64

        self.model = FNN(state_size, action_size)
        self.target_model = FNN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.train_step = 0
        self.target_update_freq = 100

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Apply rules to reduce inference
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.stack([b[0].to(self.device) for b in minibatch]).squeeze(1)
        actions = torch.tensor([b[1] for b in minibatch], device=self.device)
        rewards = torch.tensor([b[2] for b in minibatch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([b[3].to(self.device) for b in minibatch]).squeeze(1)
        dones = torch.tensor([b[4] for b in minibatch], dtype=torch.bool, device=self.device)

        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target = rewards + self.gamma * next_q_values * (~dones)

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, path="blackjack.pth"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="blackjack.pth"):
        import os
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint.get('target_model_state_dict', checkpoint['model_state_dict']))
            self.epsilon = checkpoint.get('epsilon')
            self.model.eval()
            self.target_model.eval()
            print(f"Model loaded from {path}")
        else:
            print("Model file not found. Starting with a new model.")


def encode_state(player_points, revealed_card):
    """
    Encode the game state to [player_low, player_high, dealer_low, dealer_high]
    Low counts all Aces as 1, High counts one Ace as 11 if possible
    """
    if revealed_card == 1:
        dealer_points = [1, 11]
    else:
        dealer_points = [revealed_card, revealed_card]
    
    return player_points + dealer_points


def train(agent, game, episodes=30000):
    reward_history = deque(maxlen=1000)
    
    with tqdm(total=episodes, desc="Training") as pbar:
        for _ in range(episodes):
            game.reset()

            done, reward = game.init_deal()
            if done:
                # Blackjack or Peek on initial deal is useless for training
                reward_history.append(reward)
                pbar.set_postfix({"Avg Reward": f"{sum(reward_history)/len(reward_history):.5f}"})
                pbar.update(1)
                continue
            
            upcard = game.dealer.hand[0]
            state = encode_state(game.player.points, upcard)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)

            while not done:
                action = agent.act(state)

                done, reward = game.play(action)

                next_state = encode_state(game.player.points, upcard)
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(agent.device)

                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    reward_history.append(reward)
                    pbar.set_postfix({"Avg Reward": f"{sum(reward_history)/len(reward_history):.5f}"})
                    pbar.update(1)
                    break

            agent.replay()
    
    return agent


def test(agent, game, episodes=50000):
    total_reward = 0
    agent.epsilon = 0.0   # Disable exploration
    agent.model.eval()
    agent.target_model.eval()
    
    with tqdm(total=episodes, desc="Testing") as pbar:
        for episode in range(episodes):
            game.reset()
        
            done, reward = game.init_deal()
            # Blackjack or Peek on initial deal is useless for testing
            if done:
                total_reward += reward
                pbar.set_postfix({"Avg Reward": f"{total_reward/(episode+1):.5f}"})
                pbar.update(1)
                continue
            
            upcard = game.dealer.hand[0]
            state = encode_state(game.player.points, upcard)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            
            while not done:
                action = agent.act(state)
                
                done, reward = game.play(action)
                next_state = encode_state(game.player.points, upcard)
                state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                
                if done:
                    total_reward += reward
                    pbar.set_postfix({"Avg Reward": f"{total_reward/(episode+1):.5f}"})
                    pbar.update(1)
                    break


if __name__ == "__main__":
    deck_type = Deck # InfiniteDeck CountingDeck
    game_type = EuropeGame # AmericaGame
    deck_amount = 1

    deck = deck_type(deck_amount)
    game = game_type(deck)

    state_size = 4     # 2 player points + 2 dealer points
    action_size = 2     # 2 actions: hit, stand
    agent = DQNAgent(state_size, action_size)
    
    #agent.load_model()
    train(agent, game)
    #agent.save_model()
    #agent.load_model()
    test(agent, game)