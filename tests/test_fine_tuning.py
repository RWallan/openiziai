from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from openai import OpenAI
from pydantic import ValidationError

from openiziai.fine_tuning import FineTuning


def test_init_invalid_data():
    with pytest.raises(ValidationError):
        FineTuning(client=None, task=None)


def test_size_validation_success(openai_fine_tuning, valid_task):
    mock_path = MagicMock(spec=Path)
    mock_path.stat.return_value.st_size = 100000

    fine_tuning = FineTuning(
        client=openai_fine_tuning, task=valid_task, train_file=mock_path
    )

    assert fine_tuning.train_file == mock_path


def test_size_validation_failure(openai_fine_tuning, valid_task):
    mock_path = MagicMock(spec=Path)
    mock_path.stat.return_value.st_size = 600000000

    with pytest.raises(ValidationError):
        FineTuning(
            client=openai_fine_tuning, task=valid_task, train_file=mock_path
        )


def test_upload_file_to_openai(fine_tuning):
    with patch('builtins.open', mock_open(read_data='data')):
        fine_tuning.upload_file_to_openai()

        assert fine_tuning.file_id == 'file-id'


def test_file_id_without_uploaded(fine_tuning):
    with patch('builtins.open', mock_open(read_data='data')):
        assert not fine_tuning.file_id


def test_start_fine_tuning(fine_tuning):
    with patch('builtins.open', mock_open(read_data='data')):
        fine_tuning.upload_file_to_openai().start()
        assert fine_tuning.job_id == 'job-id'


def test_job_id_without_started_fine_tuning(fine_tuning):
    with patch('builtins.open', mock_open(read_data='data')):
        assert not fine_tuning.job_id


def test_retrieve_fine_tuning_status_with_completed(fine_tuning):
    with patch('builtins.open', mock_open(read_data='data')):
        fine_tuning.upload_file_to_openai().start()
        assert fine_tuning.status == 'COMPLETED'


def test_repr_class(fine_tuning, valid_task, openai_fine_tuning):
    assert repr(fine_tuning) == (
        'FineTuning('
        f'client={openai_fine_tuning}, '
        f'train_file={fine_tuning.train_file}, '
        f'task={valid_task}, '
        f'base_model="gpt-3.5-turbo", '
        f'model=None'
        ')'
    )


def test_retrieve_fine_tuned_model(fine_tuning, valid_task):
    with patch('builtins.open', mock_open(read_data='data')):
        fine_tuning.upload_file_to_openai().start()
        assert fine_tuning.model.name == 'fine-tuned'
        assert fine_tuning.model.task == valid_task
        assert fine_tuning.model.base_model == 'gpt-3.5-turbo'
        assert fine_tuning.model.created_at


def test_start_fine_tuning_without_uploaded_file(fine_tuning):
    assert not fine_tuning.start()


def test_retrieve_status_without_job(fine_tuning):
    assert not fine_tuning.status


def test_retrieve_model_without_completed_job(valid_task):
    mock_path = MagicMock(spec=Path)
    mock_path.stat.return_value.st_size = 100000
    client = MagicMock(spec=OpenAI)
    files_mock = MagicMock(
        create=MagicMock(return_value=MagicMock(id='file-id'))
    )
    fine_tuning_mock = MagicMock(
        jobs=MagicMock(
            create=MagicMock(return_value=MagicMock(id='job-id')),
            retrieve=MagicMock(
                return_value=MagicMock(status='failed', fine_tuned_model=None)
            ),
        )
    )

    client.files = files_mock
    client.fine_tuning = fine_tuning_mock

    fine_tuning = FineTuning(
        client=client, train_file=mock_path, task=valid_task
    )

    with patch('builtins.open', mock_open(read_data='data')):
        fine_tuning.upload_file_to_openai().start()

        assert fine_tuning.status == 'FAILED'
        assert not fine_tuning.model
