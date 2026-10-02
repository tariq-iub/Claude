#!/usr/bin/env python3
"""Check docs/BIBLIOGRAPHY.csv against Crossref (title similarity, year, first author). Requires outbound HTTPS to api.crossref.org.
Writes docs/BIBLIOGRAPHY_CROSSREF.csv with the Crossref match and a match flag. Entries without a DOI are searched by title.
NOTE: in the sandbox where this scaffold was written, api.crossref.org was blocked by the egress proxy, so this script has NOT been
run; no DOI in the bibliography is machine-verified. Run it, then fix the CSV by hand."""
import csv, difflib, json, os, sys, time, urllib.parse, urllib.request

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
src = os.path.join(root, "docs", "BIBLIOGRAPHY.csv"); dst = os.path.join(root, "docs", "BIBLIOGRAPHY_CROSSREF.csv")
UA = {"User-Agent": "vpc-bibliography-check/0.1 (mailto:please-set@example.org)"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)["message"]


rows = list(csv.DictReader(open(src)))
out = []
for r in rows:
    doi = r["doi_or_id"] if r["doi_or_id"].startswith("10.") else ""
    try:
        if doi:
            m = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
        else:
            q = urllib.parse.urlencode({"query.bibliographic": r["title"], "rows": 1})
            items = get("https://api.crossref.org/works?" + q)["items"]
            m = items[0] if items else {}
        title = (m.get("title") or [""])[0]
        year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
        sim = difflib.SequenceMatcher(None, r["title"].lower(), title.lower()).ratio()
        ok = sim > 0.85 and (year is None or abs(int(r["year"]) - int(year)) <= 1)
        out.append({**r, "crossref_doi": m.get("DOI", ""), "crossref_title": title, "crossref_year": year, "title_similarity": round(sim, 3), "crossref_match": ok})
    except Exception as e:
        out.append({**r, "crossref_doi": "", "crossref_title": f"ERROR: {e}", "crossref_year": "", "title_similarity": "", "crossref_match": False})
    time.sleep(0.3)
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(f"wrote {dst}: {sum(1 for o in out if o['crossref_match'] is True)}/{len(out)} matched")
