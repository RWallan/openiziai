from unittest.mock import MagicMock

import pytest
from openai import OpenAI

from openiziai.schemas import DataDict
from openiziai.task import Task
from openiziai.tools.train_data import TrainDataTool


@pytest.fixture()
def openai_client():
    client = MagicMock(spec=OpenAI)
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
    return DataDict(data={'key': 'value'})


@pytest.fixture()
def train_data_tool(openai_client, valid_task, valid_data_dict):
    return TrainDataTool(
        client=openai_client,
        data=valid_data_dict,
        task=valid_task,
    )
