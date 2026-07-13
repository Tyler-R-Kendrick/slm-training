"""Web playground package."""

from slm_training.web.app import create_app
from slm_training.web.service import PlaygroundService

__all__ = ["PlaygroundService", "create_app"]
