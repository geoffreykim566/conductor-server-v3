"""System prompts. One file per model role."""
from app.prompts.decider import SYSTEM_PROMPT
from app.prompts.writer import WRITER_SYSTEM_PROMPT

__all__ = ["SYSTEM_PROMPT", "WRITER_SYSTEM_PROMPT"]
