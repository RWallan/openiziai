import pytest

from openiziai.agents.manager import AgentManager
from openiziai.schemas import Message


def test_initiate_agent_manager(agent, tmp_path):
    with AgentManager(agent=agent, context_store=tmp_path) as manager:
        assert manager.agent == agent
        assert manager.context.history == []
        assert manager.context.agent_model == agent.model
        assert manager.context_store == tmp_path


def test_agent_manager_enter_method_with_pre_context(agent, context):
    manager = AgentManager(agent=agent, pre_context=context.context)
    with manager as m:
        assert m.context == context.context


def test_agent_manager_exit_method(agent, tmp_path):
    with AgentManager(agent=agent, context_store=tmp_path):
        ...
    assert len(list(tmp_path.iterdir())) == 1


def test_agent_manager_prompt(agent, tmp_path):
    expected_response = (
        '{"prompt": "Test prompt", "response": "Test response"}'
    )
    with AgentManager(agent=agent, context_store=tmp_path) as manager:
        response = manager.prompt('test prompt')
        context = manager.context

    assert response.response == expected_response
    assert context.history == [
        Message(content='test prompt'),
        Message(role='assistant', content=expected_response),
    ]


def test_exist_manager_with_exception(agent, tmp_path):
    with pytest.raises(ValueError, match='error'):
        with AgentManager(agent=agent, context_store=tmp_path):
            raise ValueError('error')
    assert len(list(tmp_path.iterdir())) == 0


def test_agent_manager_with_exporter(agent, tmp_path):
    def exporter(arg):
        _arg = [i.model_dump() for i in arg]
        with open(tmp_path / 'test.txt', 'w', encoding='utf-8') as file:
            file.write(str(_arg))

    with AgentManager(
        agent=agent, context_store=tmp_path, context_exporter=exporter
    ) as m:
        m.prompt('teste')

    expected_file = tmp_path / 'test.txt'
    assert expected_file.exists()
