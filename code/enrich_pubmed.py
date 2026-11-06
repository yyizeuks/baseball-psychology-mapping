"""PubMed metadata enrichment for the screened corpus (checkpointed, resumable).

For every record in out/corpus_final.csv:
  1. esearch by exact title ([Title] field); fallback: unfielded search, accept only
     if the normalized PubMed title equals ours or shares >= 0.9 word Jaccard.
  2. efetch (batches of 100) -> PMID, DOI, journal (full + ISO abbrev), pub year/month,
     authors (full names), per-author affiliations, MeSH descriptors, publication types.
  3. elink pubmed_pubmed_citedin (batches) -> number of citing PubMed/PMC records.

Output: out/enriched_pubmed.csv (one row per corpus record, matched or not)
        out/enrich_progress.json
NCBI usage policy: <= 3 requests / second without an API key; we sleep 0.36 s.
"""
import csv, json, os, re, ssl, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
import pandas as pd

CTX = ssl.create_default_context(cafile='/root/.ccr/ca-bundle.crt')
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
HDR = {'User-Agent': 'baseball-mapping-review (mailto:yyizeuks@gmail.com)'}
OUT = 'out'
PROG = f'{OUT}/enrich_progress.json'

def get(url, retries=4):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=40, context=CTX) as r:
                return r.read()
        except Exception as e:
            time.sleep(1.5 * (a + 1))
    return None

def norm(s): return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', str(s).lower())).strip()
def jacc(a, b):
    A, B = set(norm(a).split()), set(norm(b).split())
    return len(A & B) / max(len(A | B), 1)

def esearch(term):
    u = BASE + "esearch.fcgi?db=pubmed&retmode=json&retmax=3&term=" + urllib.parse.quote(term)
    b = get(u); time.sleep(0.36)
    if not b: return []
    try: return json.loads(b)['esearchresult'].get('idlist', [])
    except Exception: return []

def esummary_titles(pmids):
    if not pmids: return {}
    u = BASE + "esummary.fcgi?db=pubmed&retmode=json&id=" + ",".join(pmids)
    b = get(u); time.sleep(0.36)
    if not b: return {}
    try:
        res = json.loads(b)['result']
        return {p: res[p].get('title', '') for p in pmids if p in res}
    except Exception: return {}

ERRATUM_LIKE = re.compile(r'^(?:author |publisher )?(?:correction|corrigendum|erratum|retraction|retracted)\b|: (?:response|reply)\.?$', re.I)

def acceptable(pt, title):
    """A PubMed candidate is accepted when its title equals the record title (normalized) or overlaps it at Jaccard >= 0.9,
    unless the candidate is a correction, erratum, retraction notice or a response letter and the record itself is not."""
    if ERRATUM_LIKE.search(str(pt).strip()) and not ERRATUM_LIKE.search(str(title).strip()): return False
    return norm(pt) == norm(title) or jacc(pt, title) >= 0.9

def match_title(title):
    """Return (pmid, how) or (None, reason)."""
    t = re.sub(r'[\[\]"]', ' ', str(title)).strip()
    if len(t) < 8: return None, 'short-title'
    ids = esearch(f'"{t}"[Title]')
    if ids:
        tt = esummary_titles(ids)
        for p, pt in tt.items():
            if acceptable(pt, title): return p, 'title-field'
    ids = esearch(t)
    if ids:
        tt = esummary_titles(ids)
        for p, pt in tt.items():
            if acceptable(pt, title): return p, 'free-text'
    return None, 'no-match'

def efetch(pmids):
    u = BASE + "efetch.fcgi?db=pubmed&retmode=xml&id=" + ",".join(pmids)
    b = get(u); time.sleep(0.36)
    out = {}
    if not b: return out
    root = ET.fromstring(b)
    for art in root.findall('.//PubmedArticle'):
        pmid = art.findtext('.//PMID')
        rec = dict(pmid=pmid)
        rec['doi'] = next((e.text for e in art.findall('.//ArticleId') if e.get('IdType') == 'doi'), '')
        j = art.find('.//Journal')
        rec['journal'] = j.findtext('Title', '') if j is not None else ''
        rec['journal_abbrev'] = j.findtext('ISOAbbreviation', '') if j is not None else ''
        rec['pub_year'] = art.findtext('.//JournalIssue/PubDate/Year', '') or art.findtext('.//ArticleDate/Year', '')
        rec['pub_month'] = art.findtext('.//JournalIssue/PubDate/Month', '') or art.findtext('.//ArticleDate/Month', '')
        if not rec['pub_year']:
            md = art.findtext('.//JournalIssue/PubDate/MedlineDate', '')
            rec['pub_year'] = md[:4]
        auths, affs = [], []
        for a in art.findall('.//AuthorList/Author'):
            ln, fn = a.findtext('LastName', ''), a.findtext('ForeName', '')
            coll = a.findtext('CollectiveName', '')
            name = f"{ln}, {fn}".strip(', ') if ln else coll
            if name: auths.append(name)
            aff = "; ".join(x.text for x in a.findall('.//AffiliationInfo/Affiliation') if x.text)
            affs.append(aff)
        rec['authors'] = " | ".join(auths)
        rec['affiliations'] = " || ".join(affs)
        rec['n_authors'] = len(auths)
        rec['mesh'] = " | ".join(m.findtext('DescriptorName', '') for m in art.findall('.//MeshHeading'))
        rec['pub_types'] = " | ".join(p.text or '' for p in art.findall('.//PublicationType'))
        rec['language'] = art.findtext('.//Language', '')
        rec['pubmed_title'] = art.findtext('.//ArticleTitle', '')
        out[pmid] = rec
    return out

def citedin(pmids):
    """elink with one id per call keeps the mapping unambiguous."""
    out = {}
    for p in pmids:
        u = BASE + f"elink.fcgi?dbfrom=pubmed&db=pubmed&linkname=pubmed_pubmed_citedin&retmode=json&id={p}"
        b = get(u); time.sleep(0.36)
        n = 0
        if b:
            try:
                ls = json.loads(b)['linksets'][0].get('linksetdbs', [])
                n = sum(len(x.get('links', [])) for x in ls if x.get('linkname') == 'pubmed_pubmed_citedin')
            except Exception: n = 0
        out[p] = n
    return out

def main(max_seconds=1500):
    t0 = time.time()
    df = pd.read_csv(f'{OUT}/corpus_final.csv')
    prog = json.load(open(PROG)) if os.path.exists(PROG) else {'match': {}, 'fetched': {}, 'cited': {}}
    # stage 1: matching
    for i, title in enumerate(df['Article Title'].astype(str)):
        k = str(i)
        if k in prog['match']: continue
        pmid, how = match_title(title)
        prog['match'][k] = [pmid, how]
        if i % 25 == 0:
            json.dump(prog, open(PROG, 'w'))
            print(f"match {i}/{len(df)} matched={sum(1 for v in prog['match'].values() if v[0])}", flush=True)
        if time.time() - t0 > max_seconds:
            json.dump(prog, open(PROG, 'w')); print("TIME BUDGET (matching)"); return
    json.dump(prog, open(PROG, 'w'))
    # stage 2: efetch
    todo = sorted({v[0] for v in prog['match'].values() if v[0] and v[0] not in prog['fetched']})
    for s in range(0, len(todo), 100):
        batch = todo[s:s+100]
        prog['fetched'].update(efetch(batch))
        json.dump(prog, open(PROG, 'w'))
        print(f"fetched {min(s+100,len(todo))}/{len(todo)}", flush=True)
        if time.time() - t0 > max_seconds: print("TIME BUDGET (fetch)"); return
    # stage 3: citations
    todo = sorted({v[0] for v in prog['match'].values() if v[0] and v[0] not in prog['cited']})
    for s in range(0, len(todo), 50):
        batch = todo[s:s+50]
        prog['cited'].update(citedin(batch))
        json.dump(prog, open(PROG, 'w'))
        print(f"cited {min(s+50,len(todo))}/{len(todo)}", flush=True)
        if time.time() - t0 > max_seconds: print("TIME BUDGET (cite)"); return
    # assemble
    rows = []
    for i in range(len(df)):
        pmid, how = prog['match'].get(str(i), [None, 'pending'])
        rec = dict(corpus_index=i, title=df.loc[i, 'Article Title'], year=int(df.loc[i, 'Publication Year']),
                   topic=int(df.loc[i, 'topic']) if 'topic' in df else None, match=how, pmid=pmid or '')
        if pmid and pmid in prog['fetched']:
            rec.update(prog['fetched'][pmid]); rec['pubmed_citations'] = prog['cited'].get(pmid, '')
        rows.append(rec)
    pd.DataFrame(rows).to_csv(f'{OUT}/enriched_pubmed.csv', index=False)
    m = sum(1 for v in prog['match'].values() if v[0])
    print(f"DONE: matched {m}/{len(df)} = {100*m/len(df):.1f}% in {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
