import pytest
from pydantic import ValidationError

from openiziai.agents.agent import Agent
from openiziai.schemas import GPTModel


def test_initiate_with_model(client, valid_task):
    gpt_model = GPTModel(
        name='model', base_model='gpt-3.5-turbo', task=valid_task
    )

    agent = Agent(client=client, model=gpt_model)

    assert agent.fine_tuned_model == 'model'
    assert valid_task.short_backstory in agent._template
    assert valid_task.goal in agent._template
    assert valid_task.role in agent._template


def test_agent_initialization_without_model_with_task(client, valid_task):
    agent = Agent(
        client=client,
        task=valid_task,
        fine_tuned_model='fine-tuned',
    )

    assert agent.fine_tuned_model == 'fine-tuned'
    assert agent._template is not None
    assert valid_task.short_backstory in agent._template
    assert valid_task.role in agent._template
    assert valid_task.goal in agent._template


def test_agent_initialization_without_model_and_task_raises_error(client):
    with pytest.raises(ValidationError):
        Agent(client=client)
