"""The knowledge base, in the decider's prompt: rendered from seed/ at import, cited
with cite_kb. Nothing is retrieved. See README.md."""
from app.kb.cite import cite
from app.kb.data import ENTRY_NAMES, REFERENCES, ROUTES
from app.kb.paths import route_path_text
from app.kb.render import KB_TEXT

__all__ = ["ENTRY_NAMES", "KB_TEXT", "REFERENCES", "ROUTES", "cite", "route_path_text"]
