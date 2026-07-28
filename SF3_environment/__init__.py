from gymnasium.envs.registration import register

register(
    id="SF3_environment/StreetFighter3-v0",
    entry_point="SF3_environment.envs:SF3Env",
)
