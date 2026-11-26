"""Khảo sát nhanh các dạng file JSON trong kho dữ liệu Localis."""
import json, collections, sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
fs = [p for p in ROOT.rglob("*.json") if "chatbot-rag" not in p.parts]
kinds, bydir, ex = collections.Counter(), collections.defaultdict(collections.Counter), {}
for f in fs:
    top = f.relative_to(ROOT).parts[0]
    try:
        d = json.loads(f.read_text(encoding="utf-8-sig"))
    except Exception:
        k = "bad"
    else:
        if isinstance(d, list):
            k = "list[%s]" % (type(d[0]).__name__ if d else "-")
        elif isinstance(d, dict):
            if "title" in d and ("rag_text" in d or "description" in d):
                k = "place" if ("lat" in d or "address" in d or "region" in d) else "titled"
                if any(x in d for x in ("start_at", "start_date", "starts_at", "event_date", "start_time")):
                    k = "event?"
            elif "places" in d: k = "has_places"
            elif "data" in d: k = "has_data"
            else: k = "dict:" + ",".join(sorted(d.keys())[:5])
        else:
            k = type(d).__name__
    kinds[k] += 1; bydir[top][k] += 1; ex.setdefault(k, str(f.relative_to(ROOT)))
print("total", len(fs))
for k, v in kinds.most_common(25): print(v, k, "|", ex[k])
print()
for t, c in bydir.items(): print(t, dict(c.most_common(6)))
