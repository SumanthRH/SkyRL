import os

# Timeout for allocating the placement group for different actors in SkyRL
SKYRL_RAY_PG_TIMEOUT = int(os.environ.get("SKYRL_RAY_PG_TIMEOUT", 30))

# export `LD_LIBRARY_PATH` to ray runtime env.
# For some reason the `LD_LIBRARY_PATH` is not exported to the worker with .env file.
SKYRL_LD_LIBRARY_PATH_EXPORT = bool(os.environ.get("SKYRL_LD_LIBRARY_PATH_EXPORT", False))
