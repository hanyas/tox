from .envs import LinearQuadratic, LinearQuadraticGaussian


def make(env_id: str):
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered environments.")

    if env_id == "LQR-v0":
        env = LinearQuadratic(step=0.01, downsampling=10, horizon=100)
    elif env_id == "LQG-v0":
        env = LinearQuadraticGaussian(step=0.01, downsampling=10, horizon=100)
    else:
        raise ValueError("Environment ID is not registered.")

    return env, env.default_params


registered_envs = [
    "LQR-v0",
    "LQG-v0",
]
