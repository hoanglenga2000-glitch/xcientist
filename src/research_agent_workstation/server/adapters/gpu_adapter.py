from __future__ import annotations

from abc import abstractmethod

from .base import Adapter
from ..schemas.evidence import ArtifactManifest
from ..schemas.run import JobStatus, RemoteJob, ResourceEstimate


class GPUAdapter(Adapter):
    provider = "gpu"

    @abstractmethod
    def list_devices(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def estimate_job(self, task_config: dict) -> ResourceEstimate:
        raise NotImplementedError

    @abstractmethod
    def submit_job(self, run_config: dict) -> RemoteJob:
        raise NotImplementedError

    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        raise NotImplementedError

    @abstractmethod
    def fetch_artifacts(self, job_id: str) -> ArtifactManifest:
        raise NotImplementedError

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        raise NotImplementedError


class LocalMockGPUAdapter(GPUAdapter):
    provider = "mock"

    def list_devices(self) -> dict:
        return {"available": False, "devices": [], "message": "GPU interface reserved; no remote GPU connected."}

    def estimate_job(self, task_config: dict) -> ResourceEstimate:
        return ResourceEstimate(self.provider, False, None, True, "Long training would require a Human Gate.")

    def submit_job(self, run_config: dict) -> RemoteJob:
        return RemoteJob("mock_gpu_unavailable", self.provider, "unavailable", {"requires_human_gate": True})

    def get_job_status(self, job_id: str) -> JobStatus:
        return "unavailable"

    def fetch_artifacts(self, job_id: str) -> ArtifactManifest:
        return ArtifactManifest("unknown", job_id, [])

    def cancel_job(self, job_id: str) -> bool:
        return True

