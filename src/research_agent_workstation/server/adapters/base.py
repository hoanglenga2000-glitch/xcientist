from __future__ import annotations

from abc import ABC


class Adapter(ABC):
    provider: str

    def status_label(self) -> str:
        return self.provider

