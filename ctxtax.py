#!/usr/bin/env python3
"""ctxtax: where did my Claude Code tokens go? Reads session logs, prints numbers only."""
import json, os, sys, statistics
from datetime import datetime, timezone

# ===== PRICES: USD per million tokens, Anthropic first-party list prices. Edit here only. =====
# Source: Anthropic model table, cached 2026-10-06; checked 2026-10-26.
# "read" = cache-read price where published; otherwise 0.1x input (marked "est").
# A model name matches the longest key it starts with; short aliases map to the current model.
# Any other model uses "default" and the report prints a warning line.
PRICES = {
    "claude-opus-5-5":   {"in": 4.0,  "out": 20.0, "read": 0.20},
    "claude-opus-5":     {"in": 5.0,  "out": 25.0, "read": 0.50},   # read est
    "claude-opus-4":     {"in": 5.0,  "out": 25.0, "read": 0.50},   # read est
    "claude-sonnet-5":   {"in": 2.0,  "out": 10.0, "read": 0.20},
    "claude-sonnet-4":   {"in": 3.0,  "out": 15.0, "read": 0.30},   # read est
    "claude-haiku-5-5":  {"in": 0.10, "out": 0.50, "read": 0.01},   # read est; prompts <=100k
    "claude-haiku-4":    {"in": 1.0,  "out": 5.0,  "read": 0.10},   # read est
    "claude-fable-5":    {"in": 10.0, "out": 50.0, "read": 0.25},
    "opus":              {"in": 4.0,  "out": 20.0, "read": 0.20},   # alias -> Opus 5.5
    "sonnet":            {"in": 2.0,  "out": 10.0, "read": 0.20},   # alias -> Sonnet 5.5
    "haiku":             {"in": 0.10, "out": 0.50, "read": 0.01},   # alias -> Haiku 5.5
    "default":           {"in": 4.0,  "out": 20.0, "read": 0.20},   # FALLBACK for unknown models
}
# ===== end of prices =====
PRICES_CHECKED = "2026-10-26"   # keep in step with the "checked" date above
VERSION = "0.2.0"
CACHE_WRITE_X = 1.25   # 5-minute cache write; 1-hour writes cost more, so this is a floor

BANDS = [("<100k", 0, 100_000), ("100-200k", 100_000, 200_000),
         ("200-400k", 200_000, 400_000), ("400k+", 400_000, float("inf"))]

# Suggestion thresholds
T_BIG_CTX = 0.20
T_MID_CTX = 0.30
T_SUB = 0.40
T_LIMITS = 3


def price_for(model):
    m = model or ""
    best = None
    for k in PRICES:
        if k != "default" and m.startswith(k) and (best is None or len(k) > len(best)):
            best = k
    return PRICES[best or "default"]


def is_known(model):
    m = model or ""
    return any(k != "default" and m.startswith(k) for k in PRICES)


def call_cost(model, u):
    p = price_for(model)
    i = u.get("input_tokens") or 0
    w = u.get("cache_creation_input_tokens") or 0
    r = u.get("cache_read_input_tokens") or 0
    o = u.get("output_tokens") or 0
    return (i * p["in"] + w * p["in"] * CACHE_WRITE_X + r * p["read"] + o * p["out"]) / 1e6


def band_of(ctx):
    for name, lo, hi in BANDS:
        if lo <= ctx < hi:
            return name
    return BANDS[-1][0]


def parse_ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def iter_files(root):
    for dp, _, fs in os.walk(root):
        for f in sorted(fs):
            if f.endswith(".jsonl"):
                yield os.path.join(dp, f)


def is_limit_text(d):
    """True if an error message mentions a limit. Text is read, never stored or printed."""
    c = (d.get("message") or {}).get("content")
    t = ""
    if isinstance(c, str):
        t = c
    elif isinstance(c, list):
        t = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
    t = t.lower()
    return "limit" in t or "429" in t


def analyse(root):
    unknown = {}        # unknown model name -> call count
    calls = {}          # message id -> record (last line wins: final usage)
    errors = []         # (file, ts, is_limit)
    stamps = {}         # file -> list of non-error timestamps
    for path in iter_files(root):
        sub_path = "subagent" in path.replace("\\", "/").lower()
        with open(path, encoding="utf8", errors="replace") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if not isinstance(d, dict):
                    continue
                ts = parse_ts(d.get("timestamp") or "")
                if d.get("type") != "assistant":
                    if ts:
                        stamps.setdefault(path, []).append(ts)
                    continue
                msg = d.get("message") or {}
                if d.get("isApiErrorMessage"):
                    errors.append((path, ts, is_limit_text(d)))
                    continue
                if ts:
                    stamps.setdefault(path, []).append(ts)
                u = msg.get("usage")
                mid = msg.get("id")
                if not u or not mid or msg.get("model") == "<synthetic>":
                    continue
                ctx = (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0) \
                    + (u.get("cache_read_input_tokens") or 0)
                side = bool(d.get("isSidechain")) or sub_path
                if not is_known(msg.get("model")):
                    unknown[msg.get("model") or "(none)"] = unknown.get(msg.get("model") or "(none)", 0) + 1
                calls[mid] = {"cost": call_cost(msg.get("model"), u), "ctx": ctx,
                              "out": u.get("output_tokens") or 0, "side": side,
                              "group": d.get("agentId") or (path if sub_path else None)}
    total = sum(c["cost"] for c in calls.values())
    n = len(calls)
    bands = {b[0]: {"calls": 0, "cost": 0.0} for b in BANDS}
    for c in calls.values():
        b = bands[band_of(c["ctx"])]
        b["calls"] += 1
        b["cost"] += c["cost"]
    for b in bands.values():
        b["call_share"] = b["calls"] / n if n else 0.0
        b["cost_share"] = b["cost"] / total if total else 0.0
    side_cost = sum(c["cost"] for c in calls.values() if c["side"])
    groups = {}
    for c in calls.values():
        if c["side"]:
            g = groups.setdefault(c["group"] or "unknown", {"cost": 0.0, "turns": 0, "out": 0})
            g["cost"] += c["cost"]
            g["turns"] += 1
            g["out"] += c["out"]
    top = sorted(groups.values(), key=lambda g: -g["cost"])[:5]
    gaps = []
    nlimit = 0
    for path, ts, lim in errors:
        if lim:
            nlimit += 1
        if ts:
            nxt = [t for t in stamps.get(path, []) if t > ts]
            if nxt:
                gaps.append(min(nxt) - ts)
    rep = {
        "calls": n, "total_cost": total,
        "bands": bands,
        "subagent": {"cost": side_cost, "share": side_cost / total if total else 0.0,
                     "calls": sum(1 for c in calls.values() if c["side"]), "top": top},
        "errors": {"api_errors": len(errors), "limit_messages": nlimit,
                   "median_gap_min": (statistics.median(gaps) / 60 if gaps else None)},
    }
    rep["unknown_models"] = unknown
    rep["advice"] = advise(rep)
    return rep


def advise(r):
    a = []
    big = r["bands"]["400k+"]["cost_share"]
    if big > T_BIG_CTX:
        a.append(f"{big:.0%} of cost is calls with 400k+ context. Start fresh sessions, or compact earlier.")
    mid = r["bands"]["200-400k"]["cost_share"]
    if big <= T_BIG_CTX and mid > T_MID_CTX:
        a.append(f"{mid:.0%} of cost is 200-400k context. Consider compacting before it grows further.")
    if r["subagent"]["share"] > T_SUB:
        a.append(f"Sub-agents are {r['subagent']['share']:.0%} of cost. Use cheaper models for gathering work.")
    if r["errors"]["limit_messages"] >= T_LIMITS:
        a.append(f"You hit limits {r['errors']['limit_messages']} times. Spread heavy work out, or trim big sessions.")
    if not a:
        a.append("Nothing stands out. Your spend looks balanced.")
    return a


def render(r):
    L = [f"ctxtax: {r['calls']} calls, ${r['total_cost']:,.2f} at list-price equivalents"]
    if r.get("unknown_models"):
        L.append(warning_line(r))
    L += ["",
         "Context tax (by size of context sent per call)",
         f"  {'band':<10}{'calls':>8}{'% calls':>9}{'cost':>12}{'% cost':>8}"]
    for k, b in r["bands"].items():
        L.append(f"  {k:<10}{b['calls']:>8}{b['call_share']:>9.1%}{b['cost']:>12,.2f}{b['cost_share']:>8.1%}")
    s = r["subagent"]
    L += ["", f"Sub-agents: ${s['cost']:,.2f} ({s['share']:.1%} of cost, {s['calls']} calls)",
          "  Top 5 sub-agent tasks:", f"  {'#':<3}{'cost':>10}{'turns':>7}{'output tokens':>15}"]
    for i, g in enumerate(s["top"], 1):
        L.append(f"  {i:<3}{g['cost']:>10,.2f}{g['turns']:>7}{g['out']:>15,}")
    e = r["errors"]
    gap = f"{e['median_gap_min']:.1f} min" if e["median_gap_min"] is not None else "n/a"
    L += ["", f"Errors: {e['api_errors']} API-error messages, {e['limit_messages']} mention a limit; median gap after: {gap}",
          "", "What to try"] + [f"  - {x}" for x in r["advice"]]
    return "\n".join(L)


def warning_line(r):
    um = r["unknown_models"]
    return (f"WARNING: {sum(um.values())} calls used models with no price ({', '.join(sorted(um))}); "
            "costs use the default price. Add them to PRICES.")


def render_share(r):
    """Short anonymised block for pasting into a GitHub thread. No paths, ids or text."""
    L = ["ctxtax summary (numbers only)", f"calls: {r['calls']}",
         "context band shares (calls | cost):"]
    for k, b in r["bands"].items():
        L.append(f"  {k}: {b['call_share']:.0%} | {b['cost_share']:.0%}")
    L += [f"sub-agent share of cost: {r['subagent']['share']:.0%}",
          f"limit hits: {r['errors']['limit_messages']}"]
    if r.get("unknown_models"):
        L.append("note: some calls used unpriced models (default price used)")
    return chr(10).join(L)


WINDOW_HOURS = 5   # assumed length of the plan's usage window


def family(model):
    m = (model or "").lower()
    for f in ("opus", "sonnet", "haiku", "fable"):
        if f in m:
            return f
    return "other"


def meter(root):
    """Anonymised limit events. Reads quota notices and usage; keeps no paths, ids or text."""
    msgs = {}    # message id -> [ts, family, in, write, read, out, cost]
    hits = {}    # (limit type, reset epoch) -> list of notice times
    for path in iter_files(root):
        with open(path, encoding="utf8", errors="replace") as fh:
            for line in fh:
                if '"usage"' not in line and '"quotaLimits"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if not isinstance(d, dict):
                    continue
                ts = parse_ts(d.get("timestamp") or "")
                q = d.get("quotaLimits")
                if isinstance(q, dict) and q.get("status") == "rejected":
                    rs = q.get("resetsAt")
                    if isinstance(rs, (int, float)) and ts is not None:
                        rs = rs / 1000 if rs > 1e11 else rs
                        hits.setdefault((q.get("rateLimitType"), rs), []).append(ts)
                    continue
                m = d.get("message")
                if not isinstance(m, dict) or ts is None:
                    continue
                u, mid = m.get("usage"), m.get("id")
                if not u or not mid or m.get("model") == "<synthetic>":
                    continue
                rec = [ts, family(m.get("model")), u.get("input_tokens") or 0,
                       u.get("cache_creation_input_tokens") or 0,
                       u.get("cache_read_input_tokens") or 0, u.get("output_tokens") or 0,
                       call_cost(m.get("model"), u)]
                old = msgs.get(mid)
                if old is None or rec[5] >= old[5]:
                    if old:
                        rec[0] = min(rec[0], old[0])
                    msgs[mid] = rec
    rows = list(msgs.values())
    events = []
    for (typ, reset), tl in sorted(hits.items(), key=lambda kv: min(kv[1])):
        if typ != "five_hour":
            continue
        first = min(tl)
        t0 = reset - WINDOW_HOURS * 3600
        tok = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}
        cost, fam = 0.0, {}
        for r in rows:
            if t0 < r[0] <= first:
                tok["input"] += r[2]
                tok["cache_write"] += r[3]
                tok["cache_read"] += r[4]
                tok["output"] += r[5]
                cost += r[6]
                fam[r[1]] = fam.get(r[1], 0.0) + r[6]
        mix = {k: round(100 * v / cost, 1) for k, v in sorted(fam.items(), key=lambda kv: -kv[1])} if cost else {}
        events.append({"date": datetime.fromtimestamp(first, timezone.utc).strftime("%Y-%m-%d"),
                       "window_hours": WINDOW_HOURS, "cost_usd": round(cost, 2),
                       "tokens": tok, "model_mix_pct": mix})
    return {"tool": "ctxtax", "version": VERSION, "prices_checked": PRICES_CHECKED,
            "price_basis": "list price, USD per million tokens", "events": events}


def render_meter(m):
    L = [f"ctxtax meter v{m['version']} | PRICES last checked {m['prices_checked']} | {len(m['events'])} events",
         "numbers only: no paths, ids, times or text"]
    for e in m["events"]:
        t = e["tokens"]
        mix = ", ".join(f"{k} {v:g}%" for k, v in e["model_mix_pct"].items()) or "n/a"
        L.append(f"{e['date']} | {e['window_hours']}h window | cost ${e['cost_usd']:,.2f} | "
                 f"in {t['input']} write {t['cache_write']} read {t['cache_read']} out {t['output']} | {mix}")
    return "\n".join(L)


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    root = args[0] if args else os.path.join(os.path.expanduser("~"), ".claude", "projects")
    if not os.path.isdir(root):
        print("Folder not found:", root)
        return 1
    if "--meter" in argv:
        m = meter(root)
        print(json.dumps(m, indent=2) if "--json" in argv else render_meter(m))
        return 0
    r = analyse(root)
    if "--share" in argv:
        print(render_share(r))
    else:
        print(json.dumps(r, indent=2) if "--json" in argv else render(r))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
