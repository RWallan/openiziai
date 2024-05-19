from unittest.mock import AsyncMock, call

import pytest
from openiziai.tools.train_data import TrainDataTool


@pytest.mark.trio()
async def test_create_examples(train_data_tool):
    sender = AsyncMock()
    n_examples = 10
    temperature = 0.5
    max_tokens = 100
    max_context_length = 8
    expected_message = {
        'messages': [
            {'role': 'system', 'content': 'Short backstory'},
            {'role': 'user', 'content': 'Test prompt'},
            {'role': 'assistant', 'content': 'Test response'},
        ]
    }
    expected_messages = 10*[expected_message]

    await train_data_tool.create_examples(
        n_examples, temperature, max_tokens, max_context_length, sender
    )

    assert sender.send.await_args_list == [call(expected_messages)]
