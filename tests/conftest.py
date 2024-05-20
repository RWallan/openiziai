import json
from unittest.mock import MagicMock

import pytest
from openai import OpenAI

from openiziai.task import Task
from openiziai.tools.train_data import TrainDataTool


@pytest.fixture()
def client():
    client = MagicMock(spec=OpenAI)
    return client


@pytest.fixture()
def openai_chat(client):
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = MagicMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content=json.dumps({
                            'prompt': 'Test prompt',
                            'response': 'Test response',
                        })
                    )
                )
            ]
        )
    )
    return client


@pytest.fixture()
def openai_fine_tuning(client):
    client.files = MagicMock()
    client.files.create.return_value.id = 'file-id'
    return client


@pytest.fixture()
def valid_task():
    return Task(
        backstory='Test backstory',
        role='Test role',
        goal='Test goal',
        short_backstory='Short backstory',
    )


@pytest.fixture()
def valid_data_dict():
    return {'data': {'key': 'value'}}


@pytest.fixture()
def train_data_tool(openai_chat, valid_task, valid_data_dict):
    return TrainDataTool(
        client=openai_chat,
        data=valid_data_dict,
        task=valid_task,
    )
