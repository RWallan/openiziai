import time
from unittest.mock import MagicMock

import pytest

from openiziai.utils import exponential_backoff


@pytest.mark.trio()
async def test_exponential_backoff_success():
    @exponential_backoff(retries=3, base_delay=0.1)
    async def func():
        return 'success'

    result = await func()
    assert result == 'success'


@pytest.mark.trio()
async def test_exponential_backoff_retries():
    mock_func = MagicMock(side_effect=Exception('Failure'))

    @exponential_backoff(retries=3, base_delay=0.1)
    async def func():
        await mock_func()

    with pytest.raises(Exception, match='Failure'):
        await func()
    expected = 3
    assert mock_func.call_count == expected


@pytest.mark.trio()
async def test_exponential_backoff_delay():
    mock_func = MagicMock(side_effect=Exception('Failure'))

    @exponential_backoff(retries=3, base_delay=0.1)
    async def func():
        await mock_func()

    start_time = time.monotonic()

    with pytest.raises(Exception, match='Failure'):
        await func()

    end_time = time.monotonic()
    elapsed_time = end_time - start_time

    expected_min_time = 0.5
    assert elapsed_time >= expected_min_time


@pytest.mark.trio()
async def test_exponential_backoff_success_after_retry():
    attempts = 0
    max_attempt = 3

    async def func():
        nonlocal attempts
        attempts += 1
        if attempts < max_attempt:
            raise Exception('Failure')
        return 'success'

    decorated_func = exponential_backoff(retries=5, base_delay=0.1)(func)

    result = await decorated_func()
    assert result == 'success'
    assert attempts == max_attempt
