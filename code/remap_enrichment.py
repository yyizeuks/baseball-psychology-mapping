"""Re-key the PubMed/MeSH enrichment to the current corpus_final.csv / screened_out.csv by normalized title.
Previously matched records keep their PMIDs; records never looked up are left for enrich_pubmed.py / enrich_mesh.py."""
import json, re, pandas as pd
def key(s): return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', str(s).lower())).strip()
old_c = pd.read_csv('/tmp/out_old/corpus_final.csv'); old_x = pd.read_csv('/tmp/out_old/screened_out.csv')
op = json.load(open('/tmp/out_old/enrich_progress.json')); om = json.load(open('/tmp/out_old/mesh_progress.json'))
lookup = {}
for i, t in enumerate(old_c['Article Title']): lookup[key(t)] = op['match'].get(str(i), [None, 'no-match'])
for i, t in enumerate(old_x['Article Title']): lookup.setdefault(key(t), om['match_excluded'].get(str(i), [None, 'no-match']))
new_c = pd.read_csv('out/corpus_final.csv'); new_x = pd.read_csv('out/screened_out.csv')
match = {str(i): lookup[key(t)] for i, t in enumerate(new_c['Article Title']) if key(t) in lookup}
match_x = {str(i): lookup[key(t)] for i, t in enumerate(new_x['Article Title']) if key(t) in lookup}
print(f'corpus {len(new_c)}: {len(match)} re-keyed ({sum(1 for v in match.values() if v[0])} with PMID), {len(new_c)-len(match)} unknown')
print(f'screened out {len(new_x)}: {len(match_x)} re-keyed ({sum(1 for v in match_x.values() if v[0])} with PMID), {len(new_x)-len(match_x)} unknown')
json.dump({'match': match, 'fetched': op['fetched'], 'cited': op['cited']}, open('out/enrich_progress.json', 'w'))
json.dump({'match_excluded': match_x, 'mesh': om['mesh'], 'trees': om['trees']}, open('out/mesh_progress.json', 'w'))
need = sorted({v[0] for v in match.values() if v[0] and v[0] not in op['fetched']}); print('PMIDs needing metadata fetch:', len(need))
