"""Cooperative pause is a control boundary, never a model/tool failure."""


class RunPaused(RuntimeError):
    failure_type = "user_paused"

    def __init__(self):
        super().__init__("user_pause_requested")
