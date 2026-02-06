import random

class Deck:
    def __init__(self, amount=2):
        self.amount = amount
        self.reset()
        
    def reset(self):
        # Create a new deck and shuffle it
        a_deck = list(range(1,10))*4 + [10]*16
        self.pile = a_deck * self.amount
        random.shuffle(self.pile)

    def deal(self):
        # Draw a card from the deck
        return self.pile.pop()

class InfiniteDeck:
    def __init__(self, amount):
        self.pile = list(range(1,10)) + [10]*4
    
    def reset(self):
        pass

    def deal(self):
        # Draw a card from the deck
        return random.choice(self.pile)

class ShuffleDeck:
    def __init__(self, amount=2, thershold=0.2, cut=True):
        self.amount = amount

        self.thers_idx = round(52 * amount * thershold)
        self.cut = cut
        
        self._create()
    
    def _create(self):
        # Create a new deck and shuffle it
        a_deck = list(range(1,10))*4 + [10]*16
        self.pile = a_deck * self.amount
        random.shuffle(self.pile)

        self.card_left = 52 * self.amount
    
    def reset(self):
        cut_idx = random.randint(1, 52 * self.amount) if self.cut else 0
        if self.card_left <= max(self.thers_idx, cut_idx):
            self._create()
    
    def deal(self):
        self.card_left -= 1
        return self.pile.pop()

class CountingDeck(ShuffleDeck):
    def _create(self):
        super()._create()
        self.card_counts = [4*self.amount for _ in range(9)] + [16*self.amount]   # 4 1-9, 16 10+

    @property
    def probabilities(self):
        return [i / self.card_left for i in self.card_counts]
    
    def update_counts(self, card):
        self.card_counts[card - 1] -= 1

class Hand:
    def __init__(self):
        self.hand = []
        self.points = [0, 0]
        
    def update(self, card):
        # Put the card to the hand and compute the points
        self.hand.append(card)
        self.points[0] += card
        self.points[1] += card if card != 1 else 11
        
        if self.points[1] > 21:
            self.points[1] = self.points[0]
    
    def display(self, role, hidden):
        if not hidden:
            # Display hand and points
            print(f"{role}'s Hand: {self.hand} | Points: ", end='')
            if self.points[0] == self.points[1]:
                print(str(self.points[0]))
            else:
                print(f"{self.points[0]}/{self.points[1]}")
        else:
            print(f"{role}'s Upcard: {self.hand[0]}")

class EuropeGame:
    """
    No peek: Player does not peek at dealer's downcard
    S17: Dealer stands on soft 17
    """
    def __init__(self, deck):
        self.deck = deck
        self.reset()
    
    def reset(self):
        self.deck.reset()
        self.multiplier = 1

        self.can_double = True
        self.can_split = True

        self.player = Hand()
        self.dealer = Hand()
    
    def init_deal(self):
        # Initial deal
        self.player.update(self.deck.deal())
        self.player.update(self.deck.deal())
        self.dealer.update(self.deck.deal())

        # Check for blackjack
        if self.player.points[1] == 21:
            self.multiplier = 1.5       # 1:1.5 for blackjack
            return self._player_stand()
        
        return False, 0

    def _player_hit(self):
        self.player.update(self.deck.deal())
        if self.player.points[0] > 21:
            return True, -1 * self.multiplier
        elif self.player.points[0] == 21:
            return True, self.multiplier
        else:
            return False, 0
    
    def _player_stand(self):
        # Dealer's turn
        player_best = self.player.points[1]
        # Hard 17 and stand on soft 17
        while self.dealer.points[1] < 17:
            self.dealer.update(self.deck.deal())
        
        # Determine the result
        dealer_best = self.dealer.points[1]
        if dealer_best > 21 or player_best > dealer_best:
            return True, 1 * self.multiplier
        elif player_best < dealer_best:
            return True, -1 * self.multiplier
        else:
            return True, 0

    def play(self, action):
        if action == 0:  # hit
            self.can_double = False
            return self._player_hit()
        elif action == 1:  # stand
            return self._player_stand()
        elif action == 2 and self.can_double:  # double
            self.can_double = False
            self.multiplier = 2
            return self._player_hit()
        elif action == 3:    # split
            raise NotImplementedError
        else:
            raise ValueError(f"Illegal action {action+1}")

class AmericaGame(EuropeGame):
    """
    Peek: Player loses immediately if dealer has blackjack on initial deal
    H17: Dealer hits on soft 17
    """
    def init_deal(self):
        done, reward = super().init_deal()
        if done:
            return done, reward
        
        # Peek at dealer's downcard if upcard is Ace or 10
        if self.dealer.hand[0] in [1, 10]:
            downcard = self.deck.deal()
            self.dealer.update(downcard)
            if self.dealer.points[1] == 21:
                return True, -1

        return False, 0
    
    def _player_stand(self):
        # Dealer's turn
        player_best = self.player.points[1]
        # Hard 17 and hit on soft 17
        while self.dealer.points[1] < 17 or (self.dealer.points[1] == 17 and self.dealer.points[0] < 17):
            self.dealer.update(self.deck.deal())
        
        # Determine the result
        dealer_best = self.dealer.points[1]
        if dealer_best > 21 or player_best > dealer_best:
            return True, 1 * self.multiplier
        elif player_best < dealer_best:
            return True, -1 * self.multiplier
        else:
            return True, 0
    

class State:
    def  __init__(self):
        self.reset()

    def reset(self):
        self.state = [0, 0, 0, 0, 0]    # win, double win, lose, double lose, tie
        self.reward = 0
    
    def display(self):
        print('Stat: Win-%i Double-%i Lose-%i Double-%i Tie-%i | Avg Reward %.2f%%' 
              %(*self.state, self.reward / sum(self.state) * 100))

    def update(self, reward, idx):
        # Update the state
        self.reward += reward
        self.state[idx] += 1

def main():
    deck_type = Deck # InfiniteDeck ShuffleDeck
    game_type = EuropeGame # AmericaGame
    deck_amount = 1

    deck = deck_type(deck_amount)
    game = game_type(deck)
    state = State()

    reward_map = {
        2 : 1,
        1 : 0,
        0 : 4,
        -1 : 2,
        -2 : 3
    }
    
    running = True
    while running:
        game.reset()
        print("\nNew Game!")

        done, reward = game.init_deal()

        game.dealer.display("Dealer", True)
        game.player.display("Player", False)

        # Player's turn
        while not done:
            try:
                action = int(input("Do you want to 1.hit, 2.stand, 3.double, 4.split?\n> "))
                done, reward = game.play(action-1)
                if action != 2:
                    game.player.display("Player", False)

            except ValueError:
                print(f"Illegal action: {action}. Please enter a valid number")
                continue

        game.dealer.display("Dealer", False)
              
        state.update(reward, reward_map[reward])
        state.display()


if __name__ == "__main__":
    main()
