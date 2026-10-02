# api

The HTTP layer. `main.py` builds the FastAPI app (lifespan opens and closes the DB pool, then middleware, then the routers). Each resource is one router in `routes/`. `/docs` is disabled.

## Files

| File | What |
|---|---|
| `main.py` | The app: lifespan, middleware, routers. `uvicorn app.api.main:app` |
| `routes/chat.py` | `POST /v3/chat`: one turn, streamed as SSE |
| `routes/register.py` | `POST /v3/register`: mint a signed conductor id |
| `routes/me.py` | `GET/PUT/DELETE /v3/me`: free-message count, profile, uninstall marker |
| `routes/ratings.py` | `POST /v3/ratings`: thumbs on a reply's event |
| `schemas.py` | `ChatRequest` and its limits; the history shape whitelist |
| `history.py` | Trimming round-tripped history; spotting a parked research turn |
| `middleware.py` | Body-size cap (before validation); 422 logging |
| `sse.py` | SSE framing |
| `deps.py` | `current_user`: verifies the `X-Conductor-Id` header (the only auth) |
| `admin/` | `/admin` KPI dashboard (basic auth): page, `/admin/stats`, `/admin/events`. SQL in `queries.py` |

## /v3/chat

**Request:**
- `message`
- `history`: exactly what the last `done` sent back
- `screenshots`: base64, this turn only
- `ax_state`: AX text, this turn only
- `research_confirm`
- `resume`

**Events, in order:**
- `status` (zero or more)
- `chunk` (writer text, streamed)
- then exactly one of:
  - `done`: `event_id`, `remaining`, `sources`, `walkthrough_steps`, `auto_run`, `history`
  - `research_prompt`: `query`, `history`
  - `error`

**Order of checks:**
1. The budget breaker returns 503.
2. A free message is claimed (402 at the limit), *before* any model call, failing closed.
3. A `resume` isn't charged again. It's checked for a parked `web_research` shape, else 422.

## Quirks & why

- **Conversation state lives in the client.** The client sends back the full raw history `respond()` produced (tool_use/tool_result included), and nothing is kept here. That's what makes multi-turn behaviour (backfill, fallbacks) work the same live as in the battery. `history.trim_history` cuts to `MAX_HISTORY_MESSAGES`, snapped to a real user turn so a `tool_result` is never orphaned.
- **The client is not a trust boundary.** History is replayed straight into the model call, so `schemas.py` whitelists exactly the shapes `pipeline/model_io.serialize_content` produces. A `tool_result.content` must be a string, because a block list could smuggle images or `cache_control`. Images and `cache_control` never appear in history (the pipeline keeps them on per-call copies).
- **Oversized screenshots are dropped, not rejected.** They're best-effort context the client sends silently. One oversized PNG (an old v0.3.0 client capturing a brushed-metal plugin skin) used to 422 every turn as "message too long".
- **422s are logged** (`[validation_422]`) with the failing field, and the response strips pydantic's input echo, which for screenshots is the whole image.
- **The body-size cap is middleware**, because uvicorn reads the body before pydantic runs. It's bounded by one turn's screenshots plus a text history; screenshots never accumulate.
- **The confidence badge is disabled:** `done.source_tier` is always `""`. The tier is a trace rule and fired on plain observations ("which tracks are muted"). It's still computed for the battery and logs.
- **Known soft spot:** history isn't signed, so a hand-built "parked" transcript gets one uncharged answer. Accepted; the free tier is a soft cap.
- **Cancelling:** when the client disconnects mid-turn (Esc), the pipeline task is cancelled. A cancelled turn still uses up its free message.
- **Admin auth fails closed:** with no `ADMIN_PASSWORD`, nobody gets in.

## Adding a route

Create `routes/<resource>.py` with `router = APIRouter()`, add it to the loop in `main.py`, and use `Depends(current_user)` for anything per-user. Rate-limited routes need a `request: Request` parameter for slowapi.
