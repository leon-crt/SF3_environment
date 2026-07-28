# Street Fighter 3 Environment
This repo contains a Reinforcement Learning Environment for the game Street Fighter 3: 3rd Strike. It can be used to train and test RL or Supervised Learning models, and it was created as part of the [mirror match project](https://github.com/leon-crt/mirror_match).
### Modes
The environment supports three modes:
- `cpu`: The model will play against an arcade CPU opponent (this mode is incomplete at the time of writing, as the opponent's inputs are not read properly).
- `free`: The game boots up normally and users can play against the model.
- `selfplay`: Inputs for both the player and opponent can be fed to the environment, making it possible to use the environment for self-play training methods.

Additionally, two render modes are supported:
- `human`: The game runs at normal speed, suitable for playing against models and testing purposes.
- `turbo`: The game runs at the highest possible speed, suitable for training purposes.

### Wrappers
This repo contains the following wrappers to transform the environment's output:
- `FlattenObservation`: This wrapper flattens the observation dictionary into a single array.


## Installation

To install your new environment, run the following commands:

```{shell}
cd SF3_environment
pip install -e .
```

### Contributing
If you would like to contribute, follow these steps:
- Fork this repository
- Clone your fork
- Set up pre-commit via `pre-commit install`

This environment was implemented following the [Gymnasium documentation](https://gymnasium.farama.org).
