# Blackjack-Deep-Q-Networks

The average reward is around -0.025 while using 1 deck of cards. Not able to double split.

i.e. lose 2.5 cents every 1 dollar.

- `train.py`: train and test the non-counting-card models
- `blackjack.py`: a rough version of Blackjack. Able to play manually

*You will indeed lose money as you play more and more, no matter how*

## Encode

The key difference of this project is the state encoder.

Each hand is encoded by two values `(low, high)`, where `low` counts all Aces as 1, and `high = low + 10` if the hand contains an Ace and `high <= 21`; Otherwise `high = low`.

This representation implicitly distinguishes soft and hard hands without introducing an explicit indicator.

In short
- `low <= high <= 21`
- `high - low ∈ {0, 10}`

## Game Rules
American
- Peek: Player loses immediately if dealer has blackjack on initial deal
- H17: Dealer hits on soft 17

European
- No peek: Player does not peek at dealer's downcard
- S17: Dealer stands on soft 17

Theoretically you have a higher win rate in Europe


## Requirements
- Pytorch
- tqdm
- Cuda

## To-Do
- [ ] Write more comments
