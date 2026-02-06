import random
import pickle
from tqdm import tqdm
from collections import deque, defaultdict

from blackjack import Deck, InfiniteDeck, ShuffleDeck, EuropeGame, AmericaGame


class TabularQAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.1
        self.batch_size = 64

        self.q_table = defaultdict(lambda: [0.0] * self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, can_double):
        # Apply rules to reduce inference
        hard = state[0][1]
        if hard < 12 and not can_double:
            return 0  # hit
        elif hard == 21:
            return 1  # stand

        if random.random() <= self.epsilon:
            if can_double:
                return random.randrange(self.action_size)
            return random.randrange(self.action_size - 1)
        
        q_values = self.q_table[state]

        return int(max(range(self.action_size-1), key=lambda a: q_values[a]))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            best_next = max(self.q_table[next_state])
            target = reward if done else reward + self.gamma * best_next
            current = self.q_table[state][action]
            self.q_table[state][action] = current + self.learning_rate * (target - current)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "q_table": dict(self.q_table),
                "epsilon": self.epsilon
            }, f)
        print(f"Model saved to {path}")

    def load_model(self, path):
        import os
        if os.path.exists(path):
            with open(path, "rb") as f:
                checkpoint = pickle.load(f)
            self.q_table = defaultdict(lambda: [0.0] * self.action_size, checkpoint["q_table"])
            self.epsilon = checkpoint.get("epsilon", 1.0)
            print(f"Model loaded from {path}")
        else:
            print("Model file not found. Starting with a new model.")


def encode_state(player_points, upcard):
    """
    Encode the game state to [player_low, player_high, upcard_low]
    Low counts all Aces as 1, High counts one Ace as 11 if possible
    """
    return tuple(player_points + [upcard])


def train(agent, game, episodes=30000):
    reward_history = deque(maxlen=10000)

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

            while not done:
                action = agent.act(state, game.can_double)

                done, reward = game.play(action)

                next_state = encode_state(game.player.points, upcard)

                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    reward_history.append(reward)
                    pbar.set_postfix({"Avg Reward": f"{sum(reward_history)/len(reward_history):.5f}"})
                    pbar.update(1)
                    break

            agent.replay()


def test(agent, game, episodes=50000):
    total_reward = 0
    agent.epsilon = 0.0  # Disable exploration

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

            while not done:
                action = agent.act(state, game.can_double)

                done, reward = game.play(action)
                next_state = encode_state(game.player.points, upcard)
                state = next_state

                if done:
                    total_reward += reward
                    pbar.set_postfix({"Avg Reward": f"{total_reward/(episode+1):.5f}"})
                    pbar.update(1)
                    break


if __name__ == "__main__":
    deck_type = Deck  # InfiniteDeck CountingDeck
    game_type = EuropeGame  # AmericaGame

    deck = deck_type()
    game = game_type(deck)

    state_size = 3     # [player_low, player_high, upcard_low]
    action_size = 3    # hit, stand, double
    agent = TabularQAgent(state_size, action_size)

    #agent.load_model("tabular_q.pkl")
    train(agent, game, episodes=300000)
    deck.amount = 6
    deck.reset()
    train(agent, game, episodes=100000)
    test(agent, game)
    #agent.save_model("tabular_q.pkl")
