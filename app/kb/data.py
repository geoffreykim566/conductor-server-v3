"""The KB and routes, loaded once from seed/ at import. Rebuild after editing seed/."""
import json
from pathlib import Path

_SEED_DIR = Path(__file__).parents[2] / "seed"

_SEED = json.loads((_SEED_DIR / "problems.json").read_text())
SOLUTION_LIST: list[dict] = _SEED["solutions"]
PROBLEM_LIST: list[dict] = _SEED["problems"]
SOLUTIONS: dict[str, dict] = {s["name"]: s for s in SOLUTION_LIST}
PROBLEMS: dict[str, dict] = {p["name"]: p for p in PROBLEM_LIST}

# Every citable name. Some buckets share their name with their only solution;
# citing one resolves as the bucket.
ENTRY_NAMES: list[str] = sorted(set(PROBLEMS) | set(SOLUTIONS))

# solution name -> [(problem name, seed_weight)], strongest bucket first
BUCKETS_OF: dict[str, list[tuple[str, int]]] = {}
for _p in PROBLEM_LIST:
    for _link in _p["solutions"]:
        BUCKETS_OF.setdefault(_link["ref"], []).append((_p["name"], _link.get("seed_weight") or 0))
for _v in BUCKETS_OF.values():
    _v.sort(key=lambda b: -b[1])

# routes.json is the only place a menu path lives. Runnable routes go to
# open_setting; `reference` entries can't run (outside Logic, or never given
# steps) -- shown to the model, never queued.
ALL_ROUTES: dict[str, dict] = json.loads((_SEED_DIR / "routes.json").read_text())
ROUTES: dict[str, dict] = {n: r for n, r in ALL_ROUTES.items() if not r.get("reference")}
REFERENCES: dict[str, dict] = {n: r for n, r in ALL_ROUTES.items() if r.get("reference")}

# A typo'd route name would otherwise only surface as a refused action at run time.
for _s in SOLUTION_LIST:
    _calls = _s.get("action") or []
    for _call in (_calls if isinstance(_calls, list) else [_calls]):
        _arg = _call.get("open_setting")
        _route = _arg.get("name") if isinstance(_arg, dict) else _arg
        if _route is not None and _route not in ROUTES:
            raise ValueError(f"{_s['name']!r} maps to open_setting {_route!r}, which isn't a runnable route")
