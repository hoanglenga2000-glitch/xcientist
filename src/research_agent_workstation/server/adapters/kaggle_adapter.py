from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

from .base import Adapter
from ..schemas.connector import CredentialStatus, DownloadResult, SubmissionResult


class KaggleAdapter(Adapter):
    provider = "kaggle"

    @abstractmethod
    def validate_credentials(self) -> CredentialStatus:
        raise NotImplementedError

    @abstractmethod
    def list_competitions(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def download_competition_data(self, competition_slug: str, target_dir: Path) -> DownloadResult:
        raise NotImplementedError

    @abstractmethod
    def submit_file(self, competition_slug: str, submission_path: Path, message: str) -> SubmissionResult:
        raise NotImplementedError

    @abstractmethod
    def get_leaderboard_status(self, competition_slug: str) -> dict:
        raise NotImplementedError


class DisabledKaggleAdapter(KaggleAdapter):
    provider = "disabled"

    def validate_credentials(self) -> CredentialStatus:
        return CredentialStatus(False, "kaggle", "Kaggle API not configured")

    def list_competitions(self) -> list:
        return []

    def download_competition_data(self, competition_slug: str, target_dir: Path) -> DownloadResult:
        return DownloadResult(False, target_dir, [], "Kaggle API not configured; use local data upload.")

    def submit_file(self, competition_slug: str, submission_path: Path, message: str) -> SubmissionResult:
        return SubmissionResult(False, True, "Kaggle submission is disabled and requires Human Gate plus credentials.")

    def get_leaderboard_status(self, competition_slug: str) -> dict:
        return {"configured": False, "message": "Kaggle API not configured"}

