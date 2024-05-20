from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from pydantic import ValidationError

from openiziai.fine_tuning import FineTuning


def test_size_validation_success(openai_fine_tuning):
    mock_path = MagicMock(spec=Path)
    mock_path.stat.return_value.st_size = 100000  # Menos de 512MB

    fine_tuning = FineTuning(client=openai_fine_tuning, train_file=mock_path)

    assert fine_tuning.train_file == mock_path


def test_size_validation_failure(openai_fine_tuning):
    mock_path = MagicMock(spec=Path)
    mock_path.stat.return_value.st_size = 600000000

    with pytest.raises(ValidationError):
        FineTuning(client=openai_fine_tuning, train_file=mock_path)


def test_upload_file_to_openai(fine_tuning):
    with patch('builtins.open', mock_open(read_data="data")):
        fine_tuning.upload_file_to_openai()

        assert fine_tuning.file_id == 'file-id'


def test_file_id_without_uploaded(fine_tuning):
    with patch('builtins.open', mock_open(read_data="data")):
        assert fine_tuning.file_id == 'file-id'

