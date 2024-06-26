"""Ferramentas gerais."""

import functools
import random

from trio import sleep


def exponential_backoff(retries: int = 64, base_delay: float = 1):
    """Implementa o método de exponential backoff para retries em funções
    assíncronas.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise e
                    delay = base_delay * (2 ** (attempt)) + random.uniform(
                        0, 1
                    )
                    attempt += 1
                    await sleep(delay)

        return wrapper

    return decorator
