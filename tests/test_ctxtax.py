import json, os, sys, tempfile, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ctxtax


def asst(mid, ctx, out=10, side=False, ts="2026-01-01T00:00:00Z", model="claude-sonnet-5-5", **kw):
    d = {"type": "assistant", "isSidechain": side, "timestamp": ts,
         "message": {"id": mid, "model": model,
                     "usage": {"input_tokens": 0, "cache_read_input_tokens": ctx,
                               "cache_creation_input_tokens": 0, "output_tokens": out}}}
    d.update(kw)
    return d


def err(text, ts):
    return {"type": "assistant", "isApiErrorMessage": True, "timestamp": ts,
            "message": {"id": "e" + ts, "model": "<synthetic>",
                        "content": [{"type": "text", "text": text}]}}


class T(unittest.TestCase):
    def run_on(self, files):
        with tempfile.TemporaryDirectory() as d:
            for name, rows in files.items():
                p = os.path.join(d, name)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w", encoding="utf8") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
            return ctxtax.analyse(d)

    def test_dedup(self):
        r = self.run_on({"a.jsonl": [asst("m1", 1000), asst("m1", 1000), asst("m2", 1000)]})
        self.assertEqual(r["calls"], 2)

    def test_bands(self):
        rows = [asst("a", 50_000), asst("b", 150_000), asst("c", 300_000), asst("d", 500_000),
                asst("e", 400_000)]
        r = self.run_on({"a.jsonl": rows})
        b = r["bands"]
        self.assertEqual([b[k]["calls"] for k in ("<100k", "100-200k", "200-400k", "400k+")], [1, 1, 1, 2])
        self.assertAlmostEqual(sum(x["cost_share"] for x in b.values()), 1.0)

    def test_cost_weights(self):
        u = {"input_tokens": 1_000_000, "cache_creation_input_tokens": 1_000_000,
             "cache_read_input_tokens": 1_000_000, "output_tokens": 0}
        self.assertAlmostEqual(ctxtax.call_cost("claude-sonnet-5-5", u), 2 * (1 + 1.25) + 0.20)

    def test_sidechain(self):
        rows = {"s.jsonl": [asst("m1", 1000), asst("m2", 1000, side=True, agentId="x"),
                            asst("m3", 1000, side=True, agentId="x")],
                "s/subagents/agent-y.jsonl": [asst("m4", 1000)]}
        r = self.run_on(rows)
        self.assertEqual(r["subagent"]["calls"], 3)
        self.assertEqual(sorted(g["turns"] for g in r["subagent"]["top"]), [1, 2])

    def test_limits(self):
        rows = [asst("m1", 10, ts="2026-01-01T00:00:00Z"),
                err("You have hit your usage limit", "2026-01-01T00:10:00Z"),
                err("Internal server error", "2026-01-01T00:20:00Z"),
                asst("m2", 10, ts="2026-01-01T01:10:00Z")]
        r = self.run_on({"a.jsonl": rows})
        self.assertEqual(r["errors"]["api_errors"], 2)
        self.assertEqual(r["errors"]["limit_messages"], 1)
        self.assertAlmostEqual(r["errors"]["median_gap_min"], (50 + 60) / 2 - 0, delta=25)
        self.assertEqual(r["calls"], 2)

    def test_advice_and_no_text(self):
        r = self.run_on({"a.jsonl": [asst("a", 900_000, side=True, agentId="q")]})
        self.assertTrue(any("400k+" in x for x in r["advice"]))
        self.assertTrue(any("Sub-agents" in x for x in r["advice"]))
        self.assertNotIn("usage limit", ctxtax.render(r))

    def test_unknown_model_warning(self):
        r = self.run_on({"a.jsonl": [asst("a", 10, model="mystery-9"), asst("b", 10)]})
        self.assertEqual(r["unknown_models"], {"mystery-9": 1})
        out = ctxtax.render(r)
        self.assertIn("WARNING", out)
        self.assertIn("mystery-9", out)
        r2 = self.run_on({"a.jsonl": [asst("b", 10)]})
        self.assertNotIn("WARNING", ctxtax.render(r2))

    def test_share_has_no_paths_or_ids(self):
        rows = {"secretproj/sess-abc123.jsonl": [asst("msg_SECRET1", 900_000, side=True, agentId="agent-XYZ"),
                                                err("You have hit your usage limit", "2026-01-01T00:10:00Z")]}
        r = self.run_on(rows)
        out = ctxtax.render_share(r)
        for bad in ("secretproj", "sess-abc123", "msg_SECRET1", "agent-XYZ", ".jsonl", "/", "\\", "@", "usage limit"):
            self.assertNotIn(bad, out)
        self.assertIn("sub-agent share", out)
        self.assertIn("limit hits: 1", out)


def quota(ts, reset, typ="five_hour"):
    return {"type": "system", "timestamp": ts,
            "quotaLimits": {"status": "rejected", "rateLimitType": typ, "resetsAt": reset}}


class MeterTests(unittest.TestCase):
    RESET = 1_790_000_000   # epoch seconds; the window starts at RESET minus 5h

    def build(self, rows):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = os.path.join(d.name, "secret-project", "sess-abc123.jsonl")
        os.makedirs(os.path.dirname(p))
        with open(p, "w", encoding="utf8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return d.name

    def iso(self, epoch):
        from datetime import datetime, timezone
        return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def rows(self):
        R, I = self.RESET, self.iso
        return [
            asst("in1", 1_000_000, out=1000, ts=I(R - 4 * 3600)),
            asst("in1", 1_000_000, out=1000, ts=I(R - 4 * 3600)),            # duplicate id
            asst("in2", 0, out=2000, ts=I(R - 3 * 3600), model="claude-opus-5-5"),
            asst("before", 9_999_999, ts=I(R - 5 * 3600 - 60)),              # before the window
            asst("after", 9_999_999, ts=I(R - 3600 + 60)),                   # after the first hit
            quota(I(R - 3600), R), quota(I(R - 3500), R), quota(I(R - 3400), R),   # 3 retries, 1 event
            quota(I(R - 100), R + 5 * 3600, typ="seven_day"),                # other type, ignored
        ]

    def test_merge_and_window(self):
        m = ctxtax.meter(self.build(self.rows()))
        self.assertEqual(len(m["events"]), 1)
        e = m["events"][0]
        self.assertEqual(e["tokens"], {"input": 0, "cache_write": 0, "cache_read": 1_000_000, "output": 3000})
        want = (1_000_000 * 0.20 + 1000 * 10.0) / 1e6 + 2000 * 20.0 / 1e6
        self.assertAlmostEqual(e["cost_usd"], want, places=2)
        self.assertEqual(e["window_hours"], 5)
        self.assertEqual(set(e["model_mix_pct"]), {"sonnet", "opus"})
        self.assertAlmostEqual(sum(e["model_mix_pct"].values()), 100, delta=0.2)

    def test_no_paths_or_ids(self):
        root = self.build(self.rows())
        m = ctxtax.meter(root)
        for out in (json.dumps(m), ctxtax.render_meter(m)):
            for bad in ("secret-project", "sess-abc123", "in1", "in2", root, ":00", "resetsAt", ".jsonl"):
                self.assertNotIn(bad, out)
        self.assertRegex(m["events"][0]["date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertIn("PRICES last checked " + ctxtax.PRICES_CHECKED, ctxtax.render_meter(m))
        self.assertEqual(m["version"], ctxtax.VERSION)

    def test_cli(self):
        import io, contextlib
        root = self.build(self.rows())
        for flags in (["--meter"], ["--meter", "--json"]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(ctxtax.main([root] + flags), 0)
            self.assertIn("2026", buf.getvalue())
        self.assertEqual(len(json.loads(buf.getvalue())["events"]), 1)


if __name__ == "__main__":
    unittest.main()
