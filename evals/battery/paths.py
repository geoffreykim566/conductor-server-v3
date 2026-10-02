"""ungrounded_paths: menu paths ("X > Y") an answer states that appear in neither the
turn's tool results nor any approved route / reference path. Reported (by the
latency harness), not asserted: a path the model knew from the KB but didn't cite
counts as ungrounded, which is the point."""
import json
import re

# A menu path as answers write it: "File > Project Settings > Audio". Items start
# with a capital or digit; later words may be lowercase connectors.
_PATH_ITEM = r"[A-Z0-9][\w&/.'’…-]*(?: (?:[A-Z0-9][\w&/.'’…-]*|&|and|of|to|in|for|as|or|with))*"
_MENU_PATH = re.compile(_PATH_ITEM + r"(?:\s*>\s*" + _PATH_ITEM + r")+")
_TRAILING = re.compile(r"(?: (?:and|of|to|in|for|as|or|with))+$")


def _norm_path(text: str) -> str:
    text = text.lower().replace("…", "").replace("...", "")
    return re.sub(r"\s*>\s*", " > ", re.sub(r"\s+", " ", text)).strip()


def _route_chains() -> list[str]:
    """Every verified path the model is shown: route menu chains, display-only
    alternatives, and reference text, each split into its " > " chains."""
    from app.kb import data, paths
    chains = []
    for name, route in data.ALL_ROUTES.items():
        for piece in re.split(r", then |\(also |\(|\)|: ", paths.route_path_text(name)):
            if " > " in piece:
                chains.append(_norm_path(piece))
        chains.append(_norm_path(route.get("desc") or ""))
    return chains


def ungrounded_paths(trace: list[dict], response_text: str) -> list[str]:
    source = _norm_path(" ".join(json.dumps(c.get("output"), ensure_ascii=False) for c in trace))
    chains = _route_chains()
    out = []
    for m in _MENU_PATH.finditer(response_text or ""):
        # Keep the sentence holding the path ("1. Open X > Y", "... > Recording. In the pane").
        found = next(p for p in re.split(r"\.\s", m.group(0)) if ">" in p).rstrip(".")
        first, *rest = _norm_path(_TRAILING.sub("", found)).split(" > ")
        # The first item can swallow sentence words ("Go to File > ..."): try every tail.
        words = first.split()
        versions = [" > ".join([" ".join(words[k:])] + rest) for k in range(len(words))]
        if not any(v in source or any(v in ch for ch in chains) for v in versions):
            out.append(versions[-1])
    return out
