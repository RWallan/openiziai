import functools
import random

from trio import sleep


def exponential_backoff(retries: int = 64, base_delay: int = 1):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            while attempt <= retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as _:
                    delay = (
                        base_delay * (2**(attempt - 1)) + random.uniform(0, 1)
                    )
                    attempt += 1
                    await sleep(delay)
        return wrapper
    return decorator
