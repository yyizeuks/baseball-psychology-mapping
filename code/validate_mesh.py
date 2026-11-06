"""Independent check of the relevance screen against NLM indexer-assigned Medical Subject Headings.

Ground truth for a PubMed-matched record: psychological content indexed by NLM =
  any MeSH descriptor in tree category F (Psychiatry and Psychology: F01-F04)  OR  any heading with the
  /psychology qualifier.  Strict version: the same restricted to major topics (asterisked headings).
Screen decision: retained vs excluded by the primary relevance screen (and, to remove any dependence on the
keyword field, a title-and-abstract-only re-application of the same rule).
Output: out/mesh_validation_results.json (numbers + ready paragraphs)
"""
import json, re, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_pipeline_v2 as P
DOMS = sorted(json.load(open('labels.json'))['domains'].keys())
K0 = int(json.load(open('out/diagnostics.json'))['K'])

OUT = 'out'
d = json.load(open(f'{OUT}/mesh_validation_data.json'))
mesh, trees = d['mesh'], d['trees']
ret_match = json.load(open(f'{OUT}/enrich_progress.json'))['match']           # corpus index -> [pmid, how]
exc_match = d['match_excluded']                                                 # screened_out index -> [pmid, how]
ret = pd.read_csv(f'{OUT}/corpus_final.csv'); exc = pd.read_csv(f'{OUT}/screened_out.csv')

def truth(pmid, major_only=False):
    heads = mesh.get(pmid)
    if heads is None: return None
    if not heads: return False
    for h in heads:
        tn = trees.get(h['ui'], []) or []
        ftree = any(t.startswith('F') for t in tn)
        psyq = any(q.lower() == 'psychology' for q, _ in h['quals'])
        if major_only:
            if (ftree and h['major']) or any(q.lower() == 'psychology' and mj for q, mj in h['quals']): return True
        elif ftree or psyq: return True
    return False

psych = P.compile_lexicon(P.PSYCH_LEXICON); proc = P.compile_lexicon(P.PROCEDURAL_LEXICON)
def ta_screen(row):
    field = (str(row['Article Title']) + ' ' + str(row['Abstract'])).lower()
    n_p, n_q = len(re.findall(psych, field)), len(re.findall(proc, field))
    return (n_p >= 2) and (n_p >= n_q)

rows = []
for i, (pmid, how) in ret_match.items():
    if pmid and pmid in mesh:
        r = ret.loc[int(i)]; rows.append(dict(pmid=pmid, screen=True, ta=ta_screen(r), truth=truth(pmid), truth_major=truth(pmid, True), n_mesh=len(mesh[pmid]), dom=r.get('dom'), topic=int(r['topic'])))
for i, (pmid, how) in exc_match.items():
    if pmid and pmid in mesh:
        r = exc.loc[int(i)]; rows.append(dict(pmid=pmid, screen=False, ta=ta_screen(r), truth=truth(pmid), truth_major=truth(pmid, True), n_mesh=len(mesh[pmid]), dom='-', topic=-1, excluded_by=r.get('_excluded_by', 'relevance')))
df = pd.DataFrame(rows)
n_matched_rows = int(len(df)); n_ret_matched = int(df.screen.sum())
df = df[df.n_mesh > 0]  # records without any MeSH (not yet indexed) carry no judgment
def metrics(dec, tru):
    tp = int((dec & tru).sum()); fp = int((dec & ~tru).sum()); fn = int((~dec & tru).sum()); tn = int((~dec & ~tru).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / max(tp + fn, 1), 3),
                specificity=round(tn / max(tn + fp, 1), 3), accuracy=round((tp + tn) / max(len(dec), 1), 3))
R = dict(n_indexed=int(len(df)), n_retained=int(df.screen.sum()), n_excluded=int((~df.screen).sum()),
         prevalence_retained=round(float(df[df.screen].truth.mean()), 3), prevalence_excluded=round(float(df[~df.screen].truth.mean()), 3),
         screen_vs_mesh=metrics(df.screen.values, df.truth.values.astype(bool)),
         screen_vs_mesh_major=metrics(df.screen.values, df.truth_major.values.astype(bool)),
         ta_screen_vs_mesh=metrics(df.ta.values.astype(bool), df.truth.values.astype(bool)),
         ta_agreement_with_full_screen=round(float((df.ta.values.astype(bool) == df.screen.values).mean()), 3),
         truth_by_domain={dd: round(float(df[(df.dom == dd)].truth.mean()), 3) for dd in DOMS},
         truth_by_topic={int(t): round(float(df[df.topic == t].truth.mean()), 3) for t in range(K0)})
# characterize false positives (retained, no psychology indexing) by topic
fp = df[df.screen & ~df.truth.astype(bool)]
R['fp_by_topic'] = {int(t): int((fp.topic == t).sum()) for t in range(K0)}
R['fp_by_domain'] = {d: int((fp.dom == d).sum()) for d in DOMS}
R['prevalence_excluded_by_reason'] = {k: round(float(v), 4) for k, v in df[~df.screen].groupby('excluded_by').truth.mean().items()}
R['n_excluded_by_reason'] = {k: int(v) for k, v in df[~df.screen].groupby('excluded_by').size().items()}
R['fp_share_domain_B'] = round(float((fp.dom == 'B').mean()), 3) if len(fp) else None
# the relevance screen is applied to topical records only: metrics restricted to retained records and records
# excluded by the relevance rule (records failing the topicality rule never reached the relevance screen)
top = df[df.screen | (df.get('excluded_by', pd.Series('relevance', index=df.index)).fillna('relevance') == 'relevance')]
R['n_topical_indexed'] = int(len(top)); R['n_topicality_failures_indexed'] = int(len(df) - len(top))
R['screen_vs_mesh_topical'] = metrics(top.screen.values, top.truth.values.astype(bool))
R['ta_screen_vs_mesh_topical'] = metrics(top.ta.values.astype(bool), top.truth.values.astype(bool))
R['n_matched_not_indexed'] = dict(all=int(n_matched_rows - len(df)), retained=int(n_ret_matched - int(df.screen.sum())))
# sport-object check: MeSH 'Baseball' heading among retained and excluded indexed records
def has_bb(pmid): return any(h['name'] == 'Baseball' for h in mesh.get(pmid, []))
df['bb'] = df['pmid'].apply(has_bb)
R['baseball_heading_retained'] = round(float(df[df.screen].bb.mean()), 3)
R['baseball_heading_excluded'] = round(float(df[~df.screen].bb.mean()), 3)
R['baseball_heading_by_domain'] = {d: round(float(df[(df.dom == d)].bb.mean()), 3) for d in DOMS}
m = R['screen_vs_mesh']; mt = R['ta_screen_vs_mesh']; mm = R['screen_vs_mesh_major']
R['methods_paragraph'] = (f"As an independent check that does not rely on the authors' judgment, the screen was compared with the Medical Subject Headings (MeSH) assigned to each PubMed-indexed record by National Library of Medicine indexers. A record was taken to have indexed psychological content when at least one of its headings belonged to MeSH tree category F (Psychiatry and Psychology) or carried the /psychology qualifier; a stricter version counted only major-topic headings. The comparison was made for every eligible record, retained or excluded, that matched a PubMed record with MeSH indexing, and the screen decision was recomputed from the title and abstract alone so that the comparison did not depend on the keyword field, which for PubMed records contains the MeSH terms themselves. Precision (the share of retained records with indexed psychological content), recall (the share of records with indexed psychological content that the screen retained), and specificity were computed. Indexing captures psychological content as a subject of the article rather than the behavioral framing that the inclusive screen was designed to admit, so the comparison characterizes the boundary of the screen rather than grading it.")
mtp = R['screen_vs_mesh_topical']; mtt = R['ta_screen_vs_mesh_topical']
R['supp_paragraph'] = (f"Independent check against MeSH indexing: {R['n_indexed']:,} eligible records with MeSH headings ({R['n_matched_not_indexed']['all']} matched records without headings omitted); psychology-category heading or /psychology qualifier present in {100*R['prevalence_retained']:.0f}% of retained and {100*R['prevalence_excluded']:.0f}% of excluded records. On the {R['n_topical_indexed']} indexed records that passed the topicality rule and therefore reached the relevance screen: precision {mtp['precision']:.2f}, recall {mtp['recall']:.2f}, specificity {mtp['specificity']:.2f}, accuracy {mtp['accuracy']:.2f} (title-and-abstract-only rule: precision {mtt['precision']:.2f}, recall {mtt['recall']:.2f}). On all {R['n_indexed']} indexed eligible records, with the records removed by the topicality rule counted as negatives: precision {m['precision']:.2f}, recall {m['recall']:.2f}, specificity {m['specificity']:.2f}, accuracy {m['accuracy']:.2f}. Per-record data are deposited (mesh_validation_data.json).")
json.dump(R, open(f'{OUT}/mesh_validation_results.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in R.items() if not k.endswith('paragraph') and k != 'response_sentence'}, indent=1))
