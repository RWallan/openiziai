import json
from pathlib import Path
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


@pytest.mark.trio()
async def test_create_train_data(train_data_tool):
    n_examples = 20
    n_batch = 2
    temperature = 0.5
    max_tokens = 100
    max_context_length = 8
    expected_len = 3

    output_file = await train_data_tool.create_train_data(
        n_examples, n_batch, temperature, max_tokens, max_context_length
    )

    assert Path(output_file).exists()
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    assert len(lines) == n_examples
    for line in lines:
        example = json.loads(line)
        assert 'messages' in example
        assert len(example['messages']) == expected_len
        assert example['messages'][1]['role'] == 'user'
        assert example['messages'][2]['role'] == 'assistant'
