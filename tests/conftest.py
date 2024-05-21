import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openai import OpenAI

from openiziai.agents import Agent
from openiziai.contexts import ContextHandler
from openiziai.fine_tuning import FineTuning
from openiziai.schemas import GPTModel
from openiziai.task import Task
from openiziai.tools.train_data import TrainDataTool


@pytest.fixture()
def client():
    client = MagicMock(spec=OpenAI)
    return client


@pytest.fixture()
def openai_chat():
    client = MagicMock(spec=OpenAI)
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = MagicMock(
        return_value=MagicMock(
            id='123',
            choices=[
                MagicMock(
                    message=MagicMock(
                        content=json.dumps({
                            'prompt': 'Test prompt',
                            'response': 'Test response',
                        })
                    )
                )
            ],
            usage=MagicMock(total_tokens=500),
        )
    )
    return client


@pytest.fixture()
def openai_fine_tuning():
    client = MagicMock(spec=OpenAI)
    files_mock = MagicMock(
        create=MagicMock(return_value=MagicMock(id='file-id'))
    )
    fine_tuning_mock = MagicMock(
        jobs=MagicMock(
            create=MagicMock(return_value=MagicMock(id='job-id')),
            retrieve=MagicMock(
                return_value=MagicMock(
                    status='succeeded', fine_tuned_model='fine-tuned'
                )
            ),
        )
    )

    client.files = files_mock
    client.fine_tuning = fine_tuning_mock
    return client


@pytest.fixture()
def fine_tuning(valid_task, openai_fine_tuning):
    mock_path = MagicMock(spec=Path)
    mock_path.stat.return_value.st_size = 100000

    fine_tuning = FineTuning(
        client=openai_fine_tuning, train_file=mock_path, task=valid_task
    )

    return fine_tuning


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


@pytest.fixture()
def context(valid_task, tmp_path):
    agent_model = GPTModel(
        name='fine-tuned', task=valid_task, base_model='gpt-3.5-turbo'
    )
    ctx = ContextHandler(
        max_context_length=3, context_store=tmp_path, agent_model=agent_model
    )
    return ctx


@pytest.fixture()
def agent(openai_chat, valid_task):
    agent = Agent(
        client=openai_chat,
        task=valid_task,
        fine_tuned_model='fine-tuned',
    )

    return agent
