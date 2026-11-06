"""Write the data-derived domain membership recorded in labels.json into the corpus files.

labels.json records, for each topic of the selected solution, the author-assigned label and the domain obtained from the
hierarchical clustering of topic centroids (analysis_sensitivity_v2.py, block 'domains'; the cut that maximizes the
document-level silhouette). This script adds the 'dom' column to out/corpus_final.csv and out/doc_assignments.csv so that
the downstream scripts (bibliometrics_v2.py, derived_stats.py, analysis_biomech_sens.py, figures_v2.py) share one source.
"""
import json, pandas as pd, os
OUT = 'out'
L = json.load(open('labels.json'))
dom = {int(k): v['domain'] for k, v in L['topics'].items()}
for f in ('corpus_final.csv', 'doc_assignments.csv'):
    p = f'{OUT}/{f}'
    if not os.path.exists(p): continue
    df = pd.read_csv(p)
    if 'topic' not in df: continue
    df['dom'] = df['topic'].map(dom)
    assert df['dom'].notna().all(), f'unlabeled topic in {f}'
    df.to_csv(p, index=False)
    print(f, 'domains:', df['dom'].value_counts().sort_index().to_dict())
