"""Independent validation data for the relevance screen.

1. Match the records excluded by the relevance screen (out/screened_out.csv) to PubMed
   (the retained corpus was matched by enrich_pubmed.py).
2. For every matched record (retained + excluded) fetch MeSH headings with descriptor UI,
   qualifiers, and major-topic flag.
3. Fetch MeSH tree numbers for every distinct descriptor (esummary db=mesh).
Output: out/mesh_validation_data.json
"""
import json, os, sys, time, ssl, urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enrich_pubmed as EP

OUT = 'out'; PROG = f'{OUT}/mesh_progress.json'

def fetch_mesh(pmids):
    u = EP.BASE + "efetch.fcgi?db=pubmed&retmode=xml&id=" + ",".join(pmids)
    b = EP.get(u); time.sleep(0.36)
    out = {}
    if not b: return out
    root = ET.fromstring(b)
    for art in root.findall('.//PubmedArticle'):
        pmid = art.findtext('.//PMID'); heads = []
        for m in art.findall('.//MeshHeading'):
            d = m.find('DescriptorName')
            heads.append(dict(ui=d.get('UI'), name=d.text, major=d.get('MajorTopicYN') == 'Y',
                              quals=[(q.text, q.get('MajorTopicYN') == 'Y') for q in m.findall('QualifierName')]))
        out[pmid] = heads
    return out

def main(max_seconds=2400):
    t0 = time.time()
    prog = json.load(open(PROG)) if os.path.exists(PROG) else {'match_excluded': {}, 'mesh': {}, 'trees': {}}
    exc = pd.read_csv(f'{OUT}/screened_out.csv')
    for i, title in enumerate(exc['Article Title'].astype(str)):
        k = str(i)
        if k in prog['match_excluded']: continue
        pmid, how = EP.match_title(title)
        prog['match_excluded'][k] = [pmid, how]
        if i % 25 == 0:
            json.dump(prog, open(PROG, 'w')); print(f"excluded match {i}/{len(exc)}", flush=True)
        if time.time() - t0 > max_seconds: json.dump(prog, open(PROG, 'w')); print('TIME BUDGET'); return
    json.dump(prog, open(PROG, 'w'))
    ret = json.load(open(f'{OUT}/enrich_progress.json'))['match']
    pm_ret = sorted({v[0] for v in ret.values() if v[0]})
    pm_exc = sorted({v[0] for v in prog['match_excluded'].values() if v[0]})
    todo = [p for p in sorted(set(pm_ret) | set(pm_exc)) if p not in prog['mesh']]
    for s in range(0, len(todo), 100):
        prog['mesh'].update(fetch_mesh(todo[s:s+100])); json.dump(prog, open(PROG, 'w'))
        print(f"mesh {min(s+100, len(todo))}/{len(todo)}", flush=True)
    uis = sorted({h['ui'] for heads in prog['mesh'].values() for h in heads if h['ui'] and h['ui'] not in prog['trees']})
    for s in range(0, len(uis), 150):
        batch = uis[s:s+150]
        ids = ",".join('68' + u[1:] for u in batch)
        b = EP.get(EP.BASE + "esummary.fcgi?db=mesh&version=2.0&retmode=json&id=" + ids); time.sleep(0.36)
        if b:
            res = json.loads(b)['result']
            for u in batch:
                rec = res.get('68' + u[1:], {})
                prog['trees'][u] = [x.get('treenum') for x in rec.get('ds_idxlinks', [])]
        json.dump(prog, open(PROG, 'w')); print(f"trees {min(s+150, len(uis))}/{len(uis)}", flush=True)
    json.dump(dict(match_excluded=prog['match_excluded'], mesh=prog['mesh'], trees=prog['trees'],
                   pmids_retained=pm_ret, pmids_excluded=pm_exc), open(f'{OUT}/mesh_validation_data.json', 'w'))
    print(f"DONE: excluded matched {len(pm_exc)}/{len(exc)}; mesh for {len(prog['mesh'])} records; trees for {len(prog['trees'])} descriptors; {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
