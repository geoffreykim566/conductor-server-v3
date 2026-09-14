"""GET /admin — internal KPI dashboard. Ported from v1 (server/app/routes/admin.py);
same page (app/static/admin.html, copied as-is), same basic-auth gate, same two
endpoints it calls. v1's /admin/kb-candidates + /admin/kb-promote (community KB
promotion) aren't ported -- admin.html never calls them, and they map to v1's
kb_entries/community_verified_at concept, which has no equivalent in v3's
solutions/problems schema (see migrations/000_schema.sql).
"""
import secrets
from datetime import date as dt_date
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app import db
from app.config import ADMIN_PASSWORD, ADMIN_USER

router = APIRouter()
_security = HTTPBasic()
_HTML = Path(__file__).parent / "static" / "admin.html"


def _require_admin(credentials: HTTPBasicCredentials = Depends(_security)) -> None:
    # Fail closed: with no password configured (e.g. env vars missing), nobody
    # gets in -- never fall through to comparing against the empty default.
    ok = bool(ADMIN_PASSWORD) and \
         secrets.compare_digest(credentials.username, ADMIN_USER) and \
         secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )


@router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(_: None = Depends(_require_admin)):
    return HTMLResponse(_HTML.read_text())


@router.get("/admin/stats", include_in_schema=False)
async def stats(_: None = Depends(_require_admin)):
    p = db.pool()

    overview = dict(await p.fetchrow("""
        WITH net AS (SELECT id FROM users WHERE uninstalled_at IS NULL)
        SELECT
            (SELECT count(*)::int FROM net)                                              AS total_users,
            (SELECT count(*)::int FROM users WHERE uninstalled_at IS NOT NULL)           AS uninstalled_users,
            (SELECT count(DISTINCT user_id)::int FROM events
              WHERE created_at > now() - interval '1 day')                               AS dau,
            (SELECT count(DISTINCT user_id)::int FROM events
              WHERE created_at > now() - interval '7 days')                              AS wau,
            (SELECT count(DISTINCT user_id)::int FROM events
              WHERE created_at > now() - interval '30 days')                             AS mau,
            (SELECT count(*)::int FROM events)                                           AS total_events,
            (SELECT count(*)::int FROM events
              WHERE created_at > now() - interval '30 days')                             AS events_30d,
            (SELECT count(*)::int FROM events
              WHERE created_at > now() - interval '1 day')                               AS events_today,
            (SELECT count(*)::int FROM users
              WHERE uninstalled_at IS NULL AND free_used >= free_limit)                  AS free_limit_hit,
            (SELECT count(*)::int FROM users
              WHERE uninstalled_at IS NULL
                AND free_used::float / NULLIF(free_limit, 0) >= 0.8)                    AS near_limit
    """))

    cost = dict(await p.fetchrow("""
        SELECT
            coalesce(round(sum(usd_cost)::numeric, 4), 0)::float                                               AS total_usd,
            coalesce(round(sum(usd_cost) FILTER (WHERE created_at > now() - interval '7 days')::numeric, 4), 0)::float
                                                                                                                AS last_7d_usd,
            coalesce(round(sum(usd_cost) FILTER (WHERE created_at > now() - interval '1 day')::numeric, 4), 0)::float
                                                                                                                AS today_usd,
            coalesce(round(
                (sum(usd_cost) FILTER (WHERE created_at > now() - interval '7 days') /
                 NULLIF(count(DISTINCT user_id) FILTER (WHERE created_at > now() - interval '7 days'), 0)
                )::numeric, 4), 0)::float                                                                       AS avg_per_user_usd
        FROM event_costs
    """))

    quality = dict(await p.fetchrow("""
        SELECT
            count(*) FILTER (WHERE rating =  1)::int                                                           AS thumbs_up,
            count(*) FILTER (WHERE rating = -1)::int                                                           AS thumbs_down,
            round(100.0 * count(*) FILTER (WHERE rating IS NOT NULL) / NULLIF(count(*), 0), 1)::float          AS rated_pct,
            round(avg(latency_ms))::int                                                                         AS avg_latency_ms,
            round((percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms))::numeric)::int                    AS p95_latency_ms
        FROM events
    """))

    experience_rows = await p.fetch("""
        SELECT coalesce(experience, 'unknown') AS label, count(*)::int AS count
        FROM users WHERE uninstalled_at IS NULL GROUP BY 1 ORDER BY 2 DESC
    """)
    role_rows = await p.fetch("""
        SELECT coalesce(role, 'unknown') AS label, count(*)::int AS count
        FROM users WHERE uninstalled_at IS NULL GROUP BY 1 ORDER BY 2 DESC
    """)
    source_rows = await p.fetch("""
        SELECT source AS label, count(*)::int AS count
        FROM events GROUP BY 1 ORDER BY 2 DESC
    """)

    # Which bucket of the free-message cap a user falls into -- subscription pricing signal.
    usage_rows = await p.fetch("""
        WITH bucketed AS (
            SELECT
                CASE
                    WHEN free_used = 0           THEN '0'
                    WHEN free_used <= 5          THEN '1-5'
                    WHEN free_used <= 10         THEN '6-10'
                    WHEN free_used <= 15         THEN '11-15'
                    WHEN free_used <= 20         THEN '16-20'
                    ELSE '20+'
                END AS bucket,
                CASE
                    WHEN free_used = 0           THEN 0
                    WHEN free_used <= 5          THEN 1
                    WHEN free_used <= 10         THEN 6
                    WHEN free_used <= 15         THEN 11
                    WHEN free_used <= 20         THEN 16
                    ELSE 21
                END AS sort_key
            FROM users WHERE uninstalled_at IS NULL
        )
        SELECT bucket, sort_key, count(*)::int AS users
        FROM bucketed
        GROUP BY bucket, sort_key
        ORDER BY sort_key
    """)

    retention = dict(await p.fetchrow("""
        SELECT
            count(*) FILTER (WHERE EXISTS (
                SELECT 1 FROM events ev
                WHERE ev.user_id = u.id
                  AND ev.created_at > u.created_at + interval '1 day'
                  AND ev.created_at <= u.created_at + interval '2 days'
            ))::int                                                                      AS came_back_d1,
            count(*) FILTER (WHERE u.created_at <= now() - interval '2 days')::int      AS d1_eligible,
            count(*) FILTER (WHERE EXISTS (
                SELECT 1 FROM events ev
                WHERE ev.user_id = u.id
                  AND ev.created_at > u.created_at + interval '7 days'
                  AND ev.created_at <= u.created_at + interval '8 days'
            ))::int                                                                      AS came_back_d7,
            count(*) FILTER (WHERE u.created_at <= now() - interval '8 days')::int      AS d7_eligible
        FROM users u
    """))
    retention["power_users_5plus"] = await p.fetchval("""
        SELECT count(*)::int FROM (
            SELECT user_id FROM events GROUP BY user_id HAVING count(*) >= 5
        ) t
        JOIN users u ON u.id = t.user_id
        WHERE u.uninstalled_at IS NULL
    """)

    events_by_day = await p.fetch("""
        SELECT
            date_trunc('day', created_at)::date::text AS day,
            count(*)::int                              AS events,
            count(DISTINCT user_id)::int               AS active_users
        FROM events
        WHERE created_at > now() - interval '30 days'
        GROUP BY 1 ORDER BY 1
    """)
    new_users_by_day = await p.fetch("""
        SELECT
            date_trunc('day', created_at)::date::text AS day,
            count(*)::int                              AS new_users
        FROM users
        WHERE created_at > now() - interval '30 days'
        GROUP BY 1 ORDER BY 1
    """)
    uninstalls_by_day = await p.fetch("""
        SELECT
            date_trunc('day', uninstalled_at)::date::text AS day,
            count(*)::int                                  AS uninstalls
        FROM users
        WHERE uninstalled_at > now() - interval '30 days'
        GROUP BY 1 ORDER BY 1
    """)
    cost_by_day = await p.fetch("""
        SELECT
            date_trunc('day', created_at)::date::text                           AS day,
            coalesce(round(sum(usd_cost)::numeric, 4), 0)::float                AS cost_usd
        FROM event_costs
        WHERE created_at > now() - interval '30 days'
        GROUP BY 1 ORDER BY 1
    """)
    cost_per_user_by_day = await p.fetch("""
        SELECT
            date_trunc('day', created_at)::date::text                                              AS day,
            coalesce(round((sum(usd_cost) / NULLIF(count(DISTINCT user_id), 0))::numeric, 6), 0)::float AS cost_per_dau
        FROM event_costs
        WHERE created_at > now() - interval '30 days'
        GROUP BY 1 ORDER BY 1
    """)
    tokens_by_day = await p.fetch("""
        SELECT
            date_trunc('day', created_at)::date::text AS day,
            sum(tokens_in)::bigint                     AS tokens_in,
            sum(tokens_out)::bigint                    AS tokens_out
        FROM events
        WHERE created_at > now() - interval '30 days'
        GROUP BY 1 ORDER BY 1
    """)
    # Only includes users who have at least one event; users who signed up but
    # never sent a message are excluded (no row in event_costs).
    cost_distribution = await p.fetch("""
        WITH user_costs AS (
            SELECT user_id, sum(usd_cost) AS total_cost
            FROM event_costs
            GROUP BY user_id
        )
        SELECT
            CASE
                WHEN total_cost = 0    THEN '$0'
                WHEN total_cost < 0.01 THEN '<$0.01'
                WHEN total_cost < 0.05 THEN '$0.01–$0.05'
                WHEN total_cost < 0.10 THEN '$0.05–$0.10'
                ELSE '$0.10+'
            END AS bucket,
            CASE
                WHEN total_cost = 0    THEN 0
                WHEN total_cost < 0.01 THEN 1
                WHEN total_cost < 0.05 THEN 2
                WHEN total_cost < 0.10 THEN 3
                ELSE 4
            END AS sort_key,
            count(*)::int AS users
        FROM user_costs
        GROUP BY bucket, sort_key
        ORDER BY sort_key
    """)

    return {
        "overview": overview,
        "cost": cost,
        "quality": quality,
        "retention": retention,
        "segments": {
            "experience": [dict(r) for r in experience_rows],
            "role":       [dict(r) for r in role_rows],
            "source":     [dict(r) for r in source_rows],
        },
        "usage_distribution": [
            {"bucket": r["bucket"], "users": r["users"]} for r in usage_rows
        ],
        "cost_distribution": [
            {"bucket": r["bucket"], "users": r["users"]} for r in cost_distribution
        ],
        "time_series": {
            "events_by_day":       [dict(r) for r in events_by_day],
            "new_users_by_day":    [dict(r) for r in new_users_by_day],
            "cost_by_day":         [dict(r) for r in cost_by_day],
            "cost_per_user_by_day":[dict(r) for r in cost_per_user_by_day],
            "tokens_by_day":       [dict(r) for r in tokens_by_day],
            "uninstalls_by_day":   [dict(r) for r in uninstalls_by_day],
        },
    }


@router.get("/admin/events", include_in_schema=False)
async def events_log(
    _: None = Depends(_require_admin),
    page: int = 1,
    limit: int = 50,
    user_id: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    event_type: str | None = None,
):
    p = db.pool()
    limit = max(1, min(limit, 200))
    offset = (max(1, page) - 1) * limit

    conditions: list[str] = []
    params: list = []

    if user_id:
        params.append(user_id)
        conditions.append(f"user_id::text = ${len(params)}")
    if from_date:
        params.append(dt_date.fromisoformat(from_date))
        conditions.append(f"created_at::date >= ${len(params)}")
    if to_date:
        params.append(dt_date.fromisoformat(to_date))
        conditions.append(f"created_at::date <= ${len(params)}")
    if event_type:
        params.append(event_type)
        conditions.append(f"type = ${len(params)}")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    total = await p.fetchval(f"SELECT count(*)::int FROM events {where}", *params)

    params.extend([limit, offset])
    rows = await p.fetch(f"""
        SELECT
            id, created_at, user_id, model, type,
            prompt, response, rating,
            tokens_in, tokens_out, latency_ms, source_tier
        FROM events
        {where}
        ORDER BY created_at DESC
        LIMIT ${len(params) - 1} OFFSET ${len(params)}
    """, *params)

    pages = max(1, (total + limit - 1) // limit)

    def row_dict(r):
        return {
            "id":          str(r["id"]),
            "created_at":  r["created_at"].isoformat() if r["created_at"] else None,
            "user_id":     str(r["user_id"]) if r["user_id"] else None,
            "model":       r["model"],
            "type":        r["type"],
            "prompt":      r["prompt"],
            "response":    r["response"],
            "rating":      r["rating"],
            "tokens_in":   r["tokens_in"],
            "tokens_out":  r["tokens_out"],
            "latency_ms":  r["latency_ms"],
            "source_tier": r["source_tier"],
        }

    return {
        "rows":  [row_dict(r) for r in rows],
        "total": total,
        "page":  page,
        "limit": limit,
        "pages": pages,
    }
