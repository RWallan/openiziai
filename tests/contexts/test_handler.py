from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from pydantic import ValidationError

from openiziai.contexts import ContextHandler
from openiziai.schemas import GPTModel, Message


def test_context_creation(valid_task):
    mock_path = MagicMock(Path)
    agent_model = GPTModel(
        name='fine-tuned', task=valid_task, base_model='gpt-3.5-turbo'
    )
    ctx = ContextHandler(
        max_context_length=3, context_store=mock_path, agent_model=agent_model
    )

    assert ctx.context.id.startswith('context-')
    assert not ctx.history
    assert ctx.context_store


def test_context_create_method(context):
    expected_error = (
        'Já existe um histórico para esse contexto.'
        'Caso queira sobescrever defina `overwrite=True`.'
    )
    context.create()
    context.add(Message(role='user', content='test'))

    with pytest.raises(Exception, match=expected_error):
        context.create()


def test_context_add_method(context):
    context.create()
    message = Message(role='user', content='Hello')
    context.add(message)
    assert context.history == [message]


def test_context_save_method(context, tmp_path):
    context.create()
    context.add(Message(content='test'))
    context.save()
    assert len(list(tmp_path.iterdir())) > 0
    assert context.context.history == [Message(content='test')]


def test_context_export_method(context):
    context.create()
    context.add(Message(role='user', content='Hello'))
    mock_exporter = Mock()
    context.export_history(mock_exporter)
    mock_exporter.assert_called_once_with(
        [Message(role='user', content='Hello')],
    )


def test_context_property(context):
    context.create()
    max_context_length = 3
    for i in range(max_context_length):
        message = Message(role='user', content=f'Message {i}')
        context.add(message)
    assert len(context.history) == max_context_length
    assert context.history[0] == Message(content='Message 0')


def test_context_init_validation():
    with pytest.raises(ValidationError):
        ContextHandler(agent=None, max_context_length=5)
