from .envs import LinearQuadratic


def make(env_id: str):
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered gymnax environments.")

    if env_id == "LQR-v0":
        env = LinearQuadratic()
    else:
        raise ValueError("Environment ID is not registered.")

    return env, env.default_params


registered_envs = [
    "LQR-v0",
]
