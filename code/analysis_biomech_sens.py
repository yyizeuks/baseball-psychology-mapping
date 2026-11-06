"""Sensitivity analysis: stricter definition of psychological relevance that excludes biomechanical studies.

Variant PC ("psychology-core screen"): the primary psychological lexicon plus a biomechanical-measurement
  lexicon on the exclusion side. Rule: psych >= 2 and psych >= procedural + biomechanical.
Variant PH ("post hoc removal"): the primary corpus with the three topics of the biomechanical domain (domain C: T1, T3, T5) removed.

For each variant the full pipeline is re-run (representation, K rule, topics, trends with four estimators),
topics are matched to the primary topics on shared documents (Hungarian assignment), and domain-level
trends are computed from the primary domain membership of the retained documents.
Output: out/sens_biomech.json
"""
import json, re, sys, os
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_pipeline_v2 as P, analysis_sensitivity_v2 as S
DOMS = sorted(json.load(open('labels.json'))['domains'].keys())
K0 = int(json.load(open('out/diagnostics.json'))['K'])

BIOMECH_LEXICON = r"""kinemat|kinetic|torque|valgus|varus|electromyograph|\bemg\b|range of motion|\brom\b|
ultrasonograph|sonograph|elastograph|laxity|biomechanic|joint load|joint moment|angular velocity|ball velocity|
pitch velocity|throwing velocity|motion capture|motion analysis|shear wave|ground reaction|trunk rotation|
external rotation|internal rotation|humeral|glenohumeral|scapul|flexor|pronator|isokinetic|muscle strength|
stiffness|ulnar collateral|medial elbow|elbow stress|shoulder stress"""

key = lambda s: re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', str(s).lower())).strip()

def build_pc():
    """Primary cascade (de-duplication, completeness, language, window, topicality) with the psychology-core rule."""
    df = pd.read_excel('corpus_raw.xlsx', sheet_name='baseball_psychology')
    df, _ = P.deduplicate(df)
    # publication-type step of the primary cascade (title patterns plus the deposited PubMed publication-type list)
    nonres = df['Article Title'].fillna('').str.contains(P.NON_RESEARCH_TITLE, regex=True, case=False)
    listed = P.load_non_research_list()
    if listed:
        nonres = nonres | df['Article Title'].map(P.norm_title).isin(listed)
    df = df[~nonres]
    ok = df['Abstract'].notna() & (df['Abstract'].astype(str).str.len() >= 200); df = df[ok]
    df = df[df['Abstract'].apply(P.is_english)]
    df = df[(df['Publication Year'] >= 1990) & (df['Publication Year'] <= 2026)]
    n_bb_title = P.baseball_mentions(df['Article Title'].fillna('').str.lower())
    n_bb_body = P.baseball_mentions((df['Abstract'].fillna('') + ' ' + df['Keywords'].fillna('')).str.lower())
    df = df[(n_bb_title >= 1) | (n_bb_body >= 2)]
    field = (df['Article Title'].fillna('') + ' ' + df['Abstract'].fillna('') + ' ' + df['Keywords'].fillna('')).str.lower()
    psych = P.count_matches(field, P.PSYCH_LEXICON); proc = P.count_matches(field, P.PROCEDURAL_LEXICON)
    bio = P.count_matches(field, BIOMECH_LEXICON)
    keep = (psych >= 2) & (psych >= proc + bio)
    out = df[keep].reset_index(drop=True)
    return out, dict(eligible=int(len(df)), retained=int(len(out)), retention=round(float(len(out)/len(df)), 4), excluded_by_biomech=int(((psych >= 2) & (psych >= proc) & ~keep).sum()))

def run_variant(dfv, prim, name):
    E, Xc, vocab = P.represent(P.make_docs(dfv))
    docfreq = np.asarray((Xc > 0).sum(0)).ravel(); idf = np.log(1 + Xc.shape[0] / (docfreq + 1)); Xbin = (Xc > 0).astype(np.int8).toarray()
    sweep = P.k_sweep(E, Xc, Xbin, docfreq, idf, 6, 12)
    mc = max(12, int(round(25 * len(dfv) / len(prim))))  # admissibility threshold scaled to corpus size
    try:
        Ksel, rule = P.select_k(sweep, min_cluster=mc)
    except ValueError:  # no admissible solution at the scaled threshold: relax to the smallest cluster observed
        mc = min(s_['min_cluster'] for s_ in sweep); Ksel, rule = P.select_k(sweep, min_cluster=mc); rule['relaxed'] = True
    res = dict(name=name, n=int(len(dfv)), rule_K=Ksel, admissible=rule['admissible'], min_cluster=mc, sweep=sweep)
    pidx = {key(t): i for i, t in enumerate(prim['Article Title'])}
    shared = np.array([key(t) in pidx for t in dfv['Article Title']])
    res['n_shared_with_primary'] = int(shared.sum())
    dom_of = np.array([prim.loc[pidx[key(t)], 'dom'] if s else '-' for t, s in zip(dfv['Article Title'], shared)])
    res['retained_by_primary_domain'] = {d: int((dom_of == d).sum()) for d in DOMS}
    res['primary_domain_sizes'] = {d: int((prim['dom'] == d).sum()) for d in DOMS}
    lab0 = np.array([prim.loc[pidx[key(t)], 'topic'] if s else -1 for t, s in zip(dfv['Article Title'], shared)])
    res['retained_by_primary_topic'] = {int(t): int((lab0 == t).sum()) for t in range(K0)}
    res['primary_topic_sizes'] = {int(t): int((prim['topic'] == t).sum()) for t in range(K0)}
    res['solutions'] = {}
    for K in sorted({Ksel, K0}):
        lab, _, coh, cent, rows = P.fit_topics(E, Xc, Xbin, docfreq, idf, vocab, dfv, K, n_init=30)
        yrs, tot, cnt = S.year_counts(dfv, lab, K); tr = S.ols_share(yrs, tot, cnt); pf = P.bh_fdr([p for _, p in tr])
        # GLM + MK
        X = sm.add_constant((yrs - yrs.mean()) / 10.0); glm, mk = [], []
        for t in range(K):
            m = sm.GLM(np.column_stack([cnt[t], tot - cnt[t]]), X, family=sm.families.Binomial()).fit(); glm.append((float(m.params[1]), float(m.pvalues[1])))
            s_, z, p, sen = S.mann_kendall(cnt[t] / np.maximum(tot, 1) * 100); mk.append((sen * 10, p))
        pg = P.bh_fdr([p for _, p in glm]); pm = P.bh_fdr([p for _, p in mk])
        # match to primary topics on shared docs
        M = np.zeros((K, K0))
        for a, b in zip(lab[shared], lab0[shared]): M[a, b] += 1
        ri, ci = linear_sum_assignment(-M); mp = {int(a): int(b) for a, b in zip(ri, ci)}
        sol = dict(ari_shared=round(adjusted_rand_score(lab0[shared], lab[shared]), 3), umass_mean=round(float(coh.mean()), 3), topics=[])
        for t in range(K):
            comp = {d: int((dom_of[lab == t] == d).sum()) for d in DOMS}
            sol['topics'].append(dict(t=t, n=int((lab == t).sum()), kw=rows[t]['keywords'].split('; ')[:7], primary_match=mp.get(t), primary_overlap=int(M[t].max()),
                                      primary_domain_composition=comp, ols_b=round(tr[t][0], 2), ols_pfdr=round(float(pf[t]), 3), ols_cls=S.classify(tr[t][0], pf[t]),
                                      glm_b=round(glm[t][0], 3), glm_pfdr=round(float(pg[t]), 3), mk_b=round(mk[t][0], 2), mk_pfdr=round(float(pm[t]), 3)))
        res['solutions'][K] = sol
        print(f"[{name}] K={K} ARI(shared)={sol['ari_shared']}")
        for x in sol['topics']: print(f"   T{x['t']} n={x['n']:3d} ->prim T{x['primary_match']} ({x['primary_overlap']}) dom{x['primary_domain_composition']} | OLS {x['ols_cls']:11s} {x['ols_b']:+.2f} p={x['ols_pfdr']:.3f} GLM p={x['glm_pfdr']:.3f} MK p={x['mk_pfdr']:.3f} | {', '.join(x['kw'][:5])}")
    # domain-level trends using primary domain membership (A and C)
    sub = dfv.assign(dom=dom_of); sub = sub[sub['Publication Year'].between(2005, 2025)]
    yrs = np.arange(2005, 2026); totd = sub.groupby('Publication Year').size().reindex(yrs, fill_value=0).values
    res['domain_trends'] = {}
    for d in DOMS:
        c = sub[sub.dom == d].groupby('Publication Year').size().reindex(yrs, fill_value=0).values
        if c.sum() < 20: continue
        prop = c / np.maximum(totd, 1) * 100; sl, _, r, p, se = stats.linregress(yrs, prop)
        res['domain_trends'][d] = dict(n=int((dom_of == d).sum()), b=round(sl * 10, 4), p=round(p, 6), s0=round(float(prop[:5].mean()), 1), s1=round(float(prop[-5:].mean()), 1))
    # BH adjustment across the domains tested in this variant
    dkeys = list(res['domain_trends'].keys())
    if dkeys:
        adjd = P.bh_fdr([res['domain_trends'][d]['p'] for d in dkeys])
        for d, q in zip(dkeys, adjd): res['domain_trends'][d]['p_fdr'] = round(float(q), 6)
    # primary-topic trends inside the variant corpus (documents keep their primary topic)
    res['primary_topic_trends_in_variant'] = {}
    for t in range(K0):
        c = sub[lab0[sub.index] == t].groupby('Publication Year').size().reindex(yrs, fill_value=0).values if False else None
    sub2 = dfv.assign(ptopic=lab0); sub2 = sub2[sub2['Publication Year'].between(2005, 2025)]
    ps = []
    for t in range(K0):
        c = sub2[sub2.ptopic == t].groupby('Publication Year').size().reindex(yrs, fill_value=0).values
        if c.sum() < 15: res['primary_topic_trends_in_variant'][t] = None; ps.append(1.0); continue
        prop = c / np.maximum(totd, 1) * 100; sl, _, r, p, se = stats.linregress(yrs, prop)
        res['primary_topic_trends_in_variant'][t] = dict(n=int((lab0 == t).sum()), b=round(sl * 10, 4), p=float(p)); ps.append(p)
    adj = P.bh_fdr(ps)
    for t in range(K0):
        if res['primary_topic_trends_in_variant'][t]: res['primary_topic_trends_in_variant'][t]['p_fdr'] = round(float(adj[t]), 6)
    print(f"[{name}] domains:", res['domain_trends']); print(f"[{name}] primary topics kept:", {t: (v['n'], v['b'], v['p_fdr']) for t, v in res['primary_topic_trends_in_variant'].items() if v})
    return res

if __name__ == '__main__':
    prim = pd.read_csv('out/corpus_final.csv')
    pc, casc = build_pc(); print("PC screen:", casc)
    out = {'pc_cascade': casc}
    out['PC'] = run_variant(pc, prim, 'psychology-core screen')
    BIO_DOM = [d for d, v in json.load(open('labels.json'))['domains'].items() if 'biomechanics' in v['label'].lower()][0]
    out['bio_domain'] = BIO_DOM
    ph = prim[prim['dom'] != BIO_DOM].reset_index(drop=True)
    out['PH'] = run_variant(ph, prim, f'post hoc removal of domain {BIO_DOM}')
    json.dump(out, open('out/sens_biomech.json', 'w'), indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
    print('saved out/sens_biomech.json')
