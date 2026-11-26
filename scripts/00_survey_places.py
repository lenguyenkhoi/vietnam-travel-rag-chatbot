import json, collections, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
recs = []
for f in ROOT.rglob("*.json"):
    if "chatbot-rag" in f.parts: continue
    try: d = json.loads(f.read_text(encoding="utf-8-sig"))
    except Exception: continue
    if isinstance(d, dict) and "title" in d and ("rag_text" in d or "description" in d) and ("lat" in d or "address" in d or "region" in d):
        recs.append((f.relative_to(ROOT), d))
print("records", len(recs))
slugs = collections.Counter(d.get("slug") for _, d in recs)
print("unique slugs", len(slugs), "no-slug", slugs[None])
titles = collections.Counter(str(d.get("title_norm") or d.get("title") or "").lower() for _, d in recs)
print("unique titles", len(titles))
keys = collections.Counter(k for _, d in recs for k in d)
print(keys.most_common(70))
cat = collections.Counter(); tag0 = collections.Counter(); region = collections.Counter(); lang = collections.Counter()
for p, d in recs:
    cat[str(d.get("category"))[:40]] += 1
    t = d.get("tags") or []
    tag0[t[0] if t else None] += 1
    region[d.get("region")] += 1
    lang[d.get("lang") or d.get("language")] += 1
print("category", cat.most_common(15)); print("tag0", tag0.most_common(30)); print("region", region.most_common(40)); print("lang", lang.most_common(5))
seg = collections.Counter()
for p, d in recs:
    for s in p.parts[:-1]:
        if s.startswith("_") or s.lower() in ("old","backup","ban-cu","en","dich"): seg[s] += 1
print("underscore/odd segments", seg.most_common(30))
rl = sorted(len(d.get("rag_text") or "") for _, d in recs); print("rag_text len pct", [rl[int(len(rl)*q)] for q in (0,.1,.25,.5,.75,.9,.99)])
dl = sorted(len(d.get("description") or "") for _, d in recs); print("desc len pct", [dl[int(len(dl)*q)] for q in (0,.1,.25,.5,.75,.9,.99)])
