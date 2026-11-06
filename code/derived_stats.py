"""Derived statistics used by the manuscript text (reproducible companion to the sensitivity blocks).

Outputs (out/):
  domain_trends.json   domain-level share trends 2005-2025 under four estimators (OLS, binomial GLM, CLR, Mann-Kendall),
                       BH-adjusted within estimator; era shares; within-domain share trend of each topic
  strand_trends.json   k = 3 sub-clusters of the two largest topics: size, leading terms, median year, pre-2010 share,
                       OLS slope of the strand's annual share of the corpus (BH-adjusted within the six strands)
  label_checks.json    document-level checks used in the Discussion (country mentions by topic, Japanese affiliations,
                       era counts, composition of the 20 most cited articles)
  venue_core.json      venue-concentration indicators: share of each domain's PubMed-matched articles in its own five
                       most productive journals and in the other domains' five most productive journals
  source_check.json    share trends estimated separately within PubMed-matched and non-matched records; venue-novelty proxy
  representative_documents.csv  the 15 articles closest to each topic centroid (the evidence behind each topic label)
"""
import json, re, os, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
import statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_pipeline_v2 as P, analysis_sensitivity_v2 as S

OUT = 'out'; SEED = 42
L = json.load(open('labels.json')); TOP = {int(k): v for k, v in L['topics'].items()}; DOM = L['domains']; DOMS = sorted(DOM)
df = pd.read_csv(f'{OUT}/corpus_final.csv'); K = int(df['topic'].max()) + 1
E = np.load(f'{OUT}/E.npy')
yrs = np.arange(2005, 2026)

def four_estimators(groups, names):
    """groups: dict name -> boolean mask over df rows. Returns per-group trend statistics on annual share."""
    sub = df[df['Publication Year'].isin(yrs)]
    tot = sub.groupby('Publication Year').size().reindex(yrs, fill_value=0).values
    cnt = np.vstack([sub[groups[g].loc[sub.index]].groupby('Publication Year').size().reindex(yrs, fill_value=0).values for g in names])
    X = sm.add_constant((yrs - yrs.mean()) / 10.0)
    comp = (cnt + 0.5) / (tot + 0.5 * len(names)); logg = np.log(comp); clr_mat = logg - logg.mean(0, keepdims=True)
    ols, glm, clr, mk = [], [], [], []
    for i in range(len(names)):
        prop = cnt[i] / np.maximum(tot, 1) * 100
        sl, _, r, p, se = stats.linregress(yrs, prop); ols.append((sl * 10, p, r ** 2, se * 10))
        m = sm.GLM(np.column_stack([cnt[i], tot - cnt[i]]), X, family=sm.families.Binomial()).fit(); glm.append((float(m.params[1]), float(m.pvalues[1])))
        sl2, _, _, p2, _ = stats.linregress(yrs, clr_mat[i]); clr.append((sl2 * 10, p2))
        s_, z, p3, sen = S.mann_kendall(prop); mk.append((sen * 10, p3))
    pf = {k: P.bh_fdr([p for _, p, *_ in v]) if k == 'ols' else P.bh_fdr([p for _, p in v]) for k, v in dict(ols=ols, glm=glm, clr=clr, mk=mk).items()}
    out = {}
    for i, g in enumerate(names):
        prop = cnt[i] / np.maximum(tot, 1) * 100
        out[g] = dict(n=int(groups[g].sum()), ols_b=round(ols[i][0], 2), ols_se=round(ols[i][3], 2), ols_p=round(ols[i][1], 6), ols_pfdr=round(float(pf['ols'][i]), 6), r2=round(ols[i][2], 3),
                      glm_b=round(glm[i][0], 3), glm_p=round(glm[i][1], 6), glm_pfdr=round(float(pf['glm'][i]), 6),
                      clr_b=round(clr[i][0], 3), clr_p=round(clr[i][1], 6), clr_pfdr=round(float(pf['clr'][i]), 6),
                      mk_sen=round(mk[i][0], 2), mk_p=round(mk[i][1], 6), mk_pfdr=round(float(pf['mk'][i]), 6),
                      share_2005_09=round(float(prop[:5].mean()), 1), share_2021_25=round(float(prop[-5:].mean()), 1),
                      n_sig=int(sum(1 for k in ('ols', 'glm', 'clr', 'mk') if pf[k][i] < .05)),
                      cls=S.classify(ols[i][0], pf['ols'][i]))
    return out

# ---------------------------------------------------------------- domain-level trends
groups = {d: df['dom'] == d for d in DOMS}
dt = four_estimators(groups, DOMS)
for d in DOMS: dt[d]['name'] = DOM[d]['label']; dt[d]['topics'] = DOM[d]['topics']
# within-domain reorganization: each topic's share of its own domain (OLS + binomial GLM), for domains with > 1 topic
within = {}
for d in DOMS:
    ts = DOM[d]['topics']
    if len(ts) < 2: continue
    sub = df[(df['dom'] == d) & df['Publication Year'].isin(yrs)]
    totd = sub.groupby('Publication Year').size().reindex(yrs, fill_value=0).values
    X = sm.add_constant((yrs - yrs.mean()) / 10.0)
    for t in ts:
        c = sub[sub.topic == t].groupby('Publication Year').size().reindex(yrs, fill_value=0).values
        prop = c / np.maximum(totd, 1) * 100
        sl, _, r, p, se = stats.linregress(yrs, prop)
        m = sm.GLM(np.column_stack([c, totd - c]), X, family=sm.families.Binomial()).fit()
        within[f'{d}:T{t}'] = dict(domain=d, topic=t, ols_b=round(sl * 10, 2), ols_p=round(p, 6), glm_b=round(float(m.params[1]), 3), glm_p=round(float(m.pvalues[1]), 6),
                                   share_2005_09=round(float(prop[:5].mean()), 1), share_2021_25=round(float(prop[-5:].mean()), 1))
# BH adjustment across the distinct within-domain tests (separately for OLS and GLM): in a two-topic domain the second
# topic's test is the mirror image of the first (b = -b, identical p) and enters the family once
wk = list(within.keys()); fam = []
seen = set()
for k in wk:
    d = within[k]['domain']
    if len(DOM[d]['topics']) == 2 and d in seen: continue
    fam.append(k); seen.add(d)
if wk:
    for est in ('ols', 'glm'):
        adj_w = P.bh_fdr([within[k][f'{est}_p'] for k in fam])
        qmap = dict(zip(fam, adj_w))
        for k in wk:
            d = within[k]['domain']
            q = qmap.get(k, qmap[next(kk for kk in fam if within[kk]['domain'] == d)])
            within[k][f'{est}_pfdr'] = round(float(q), 6)
dt['within_domain_family_size'] = len(fam)
dt['within_domain'] = within
# window-start sensitivity: domain-level OLS share trends with the window starting in 2001 and in 2010 (BH within four domains)
def _dom_window(y0):
    yy = np.arange(y0, 2026); subw = df[df['Publication Year'].isin(yy)]; totw = subw.groupby('Publication Year').size().reindex(yy, fill_value=0).values
    res = {}; ps = []; pg = []; Xw = sm.add_constant((yy - yy.mean()) / 10.0)
    for d in DOMS:
        c = subw[subw['dom'] == d].groupby('Publication Year').size().reindex(yy, fill_value=0).values
        sl, _, r, p, se = stats.linregress(yy, c / np.maximum(totw, 1) * 100)
        mg = sm.GLM(np.column_stack([c, totw - c]), Xw, family=sm.families.Binomial()).fit()
        res[d] = dict(b=round(sl * 10, 2), p=round(p, 6), glm_b=round(float(mg.params[1]), 3), glm_p=round(float(mg.pvalues[1]), 6)); ps.append(p); pg.append(float(mg.pvalues[1]))
    for d, q, qg in zip(DOMS, P.bh_fdr(ps), P.bh_fdr(pg)): res[d]['p_fdr'] = round(float(q), 6); res[d]['glm_pfdr'] = round(float(qg), 6)
    return dict(window=[int(y0), 2025], n_years=int(len(yy)), n_articles=int(len(subw)), min_year_count=int(totw.min()), max_year_count_early=int(totw[:4].max()), domains=res)
dt['window_sensitivity'] = {str(y0): _dom_window(y0) for y0 in (2001, 2010)}
print('window sensitivity:', {k: {d: (v['domains'][d]['b'], v['domains'][d]['p_fdr']) for d in DOMS} for k, v in dt['window_sensitivity'].items()})
# era shares by domain
eras = [(1990, 1999), (2000, 2009), (2010, 2014), (2015, 2019), (2020, 2026)]
dt['era_share_pct'] = {d: [round(100 * float(((df['dom'] == d) & df['Publication Year'].between(a, b)).sum() / max(df['Publication Year'].between(a, b).sum(), 1)), 1) for a, b in eras] for d in DOMS}
dt['eras'] = [f'{a}-{b}' for a, b in eras]
json.dump(dt, open(f'{OUT}/domain_trends.json', 'w'), indent=1)
print('domain trends:', {d: (dt[d]['ols_b'], dt[d]['ols_pfdr'], dt[d]['n_sig']) for d in DOMS})
print('within-domain:', {k: (v['ols_b'], v['ols_p'], v['glm_p']) for k, v in within.items()})

# ---------------------------------------------------------------- strand-level trends (two largest topics, k = 3)
docs = P.make_docs(df); _, Xc, vocab = P.represent(docs)
docfreq = np.asarray((Xc > 0).sum(0)).ravel(); idf = np.log(1 + Xc.shape[0] / (docfreq + 1))
sizes = np.bincount(df['topic'].values); big = [int(t) for t in np.argsort(-sizes)[:2]]
strands = {}; pvals = []; keys = []
sub_all = df[df['Publication Year'].isin(yrs)]
tot = sub_all.groupby('Publication Year').size().reindex(yrs, fill_value=0).values
for t in big:
    m = (df['topic'] == t).values
    lab = KMeans(3, n_init=20, random_state=SEED).fit_predict(E[m])
    full = np.full(len(df), -1); full[m] = lab
    for s_ in range(3):
        ms = full == s_
        order = np.argsort((np.asarray(Xc[ms].sum(0)).ravel() / max(ms.sum(), 1)) * idf)[::-1][:8]
        c = df[ms & df['Publication Year'].isin(yrs)].groupby('Publication Year').size().reindex(yrs, fill_value=0).values
        prop = c / np.maximum(tot, 1) * 100
        sl, _, r, p, se = stats.linregress(yrs, prop)
        y = df.loc[ms, 'Publication Year']
        strands[f'T{t}s{s_}'] = dict(topic=t, strand=s_, n=int(ms.sum()), kw=[str(w) for w in vocab[order][:6]], b=round(sl * 10, 2), p=round(p, 6),
                                     median_year=int(y.median()), pre2010=int((y < 2010).sum()), share_pre2010=round(float((y < 2010).mean()), 2))
        pvals.append(p); keys.append(f'T{t}s{s_}')
adj = P.bh_fdr(pvals)
for k_, a in zip(keys, adj): strands[k_]['p_fdr'] = round(float(a), 3)
strands['_big_topics'] = big
json.dump(strands, open(f'{OUT}/strand_trends.json', 'w'), indent=1)
print('strands:', {k: (v['n'], v['kw'][:3], v['b'], v['p']) for k, v in strands.items() if not k.startswith('_')})

# ---------------------------------------------------------------- label checks
field = (df['Article Title'].fillna('') + ' ' + df['Abstract'].fillna('') + ' ' + df['Keywords'].fillna('')).str.lower()
mentions = {str(t): {c: int(field[df.topic == t].str.contains(pat).sum()) for c, pat in [('korea', r'korea'), ('japan', r'japan'), ('taiwan', r'taiwan')]} for t in range(K)}
en = pd.read_csv(f'{OUT}/enriched_pubmed.csv')
aff = en['affiliations'].fillna('').str.lower()
japan_aff = {str(t): [int((aff[(en.topic == t)].str.contains('japan')).sum()), int((aff[(en.topic == t)].str.len() > 0).sum())] for t in range(K)}
era_counts = {str(t): [int(((df.topic == t) & df['Publication Year'].between(a, b)).sum()) for a, b in eras] for t in range(K)}
tc = pd.read_csv(f'{OUT}/table_top_cited.csv') if os.path.exists(f'{OUT}/table_top_cited.csv') else None
top_cited_by_domain = {str(d): int((tc['domain'] == d).sum()) for d in DOMS} if tc is not None else {}
# baseball-centrality of the retained corpus: baseball term in the title / number of mentions
bb_title = P.baseball_mentions(df['Article Title'].fillna('').str.lower())
bb_body = P.baseball_mentions((df['Abstract'].fillna('') + ' ' + df['Keywords'].fillna('')).str.lower())
centrality = dict(title_mention_share=round(float((bb_title >= 1).mean()), 3), median_body_mentions=float(np.median(bb_body)),
                  n_title_mention=int((bb_title >= 1).sum()), n_title0_body2=int(((bb_title == 0) & (bb_body == 2)).sum()), n_title0_body_ge3=int(((bb_title == 0) & (bb_body >= 3)).sum()),
                  by_topic_title_share={str(t): round(float((bb_title[df.topic == t] >= 1).mean()), 2) for t in range(K)},
                  by_topic_title0_body2={str(t): int(((bb_title == 0) & (bb_body == 2) & (df.topic == t)).sum()) for t in range(K)})
# abstract-missing exclusions: psychological term in the title
raw = pd.read_excel('corpus_raw.xlsx', sheet_name='baseball_psychology'); raw, _ = P.deduplicate(raw)
_nonres = raw['Article Title'].fillna('').str.contains(P.NON_RESEARCH_TITLE, regex=True, case=False) | raw['Article Title'].map(P.norm_title).isin(P.load_non_research_list())
raw = raw[~_nonres]
miss = raw[~(raw['Abstract'].notna() & (raw['Abstract'].astype(str).str.len() >= 200))]
miss_psych = P.count_matches(miss['Article Title'].fillna('').str.lower(), P.PSYCH_LEXICON)
miss_bb = P.baseball_mentions(miss['Article Title'].fillna('').str.lower())
abstract_missing = dict(n=int(len(miss)), title_with_psych_term=int((miss_psych >= 1).sum()), title_with_baseball_term=int((miss_bb >= 1).sum()),
                        title_with_both=int(((miss_psych >= 1) & (miss_bb >= 1)).sum()))
# topicality exclusions: how many mention baseball exactly once (and never in the title); how many name another sport or a multi-sport sample
so = pd.read_csv(f'{OUT}/screened_out.csv'); tp = so[so['_excluded_by'] == 'topicality']
tp_txt = (tp['Article Title'].fillna('') + ' ' + tp['Abstract'].fillna('') + ' ' + tp['Keywords'].fillna('')).str.lower()
OTHER_SPORT = (r'\b(?:football|soccer|basketball|hockey|tennis|golf|volleyball|cricket|rugby|lacrosse|swimming|wrestling|gymnastics|handball|badminton|'
               r'athletes? from|multiple sports|various sports|different sports|\d+ sports)\b')
topicality = dict(n=int(len(tp)), single_mention=int(((tp['_n_bb_title'] == 0) & (tp['_n_bb_body'] == 1)).sum()),
                  other_sport_share=round(float(tp_txt.str.contains(OTHER_SPORT, regex=True).mean()), 4))
# topicality exclusions that concern perception-action research (candidate evidence for migration of perceptual-motor work into
# sport-general venues where baseball is incidental): counts by era
PERC = r'\b(?:percept|perceptual|anticipat|visual|vision|gaze|eye movement|motor control|motor skill|interception|reaction time|batting task|hitting task)'
tp_perc = tp_txt.str.contains(PERC, regex=True)
_eras = [(1990, 2009), (2010, 2014), (2015, 2019), (2020, 2026)]
topicality['perception_action_by_era'] = {f'{a}-{b}': int((tp_perc & tp['Publication Year'].between(a, b)).sum()) for a, b in _eras}
topicality['perception_action_total'] = int(tp_perc.sum())
json.dump(dict(mentions=mentions, japan_aff=japan_aff, era_counts=era_counts, top_cited_by_domain=top_cited_by_domain, centrality=centrality,
               abstract_missing=abstract_missing, topicality=topicality), open(f'{OUT}/label_checks.json', 'w'), indent=1)
print('checks:', mentions, japan_aff, top_cited_by_domain, centrality, abstract_missing)

# ---------------------------------------------------------------- venue concentration
enm = en[en['match'] != 'no-match'].copy(); enm['dom'] = enm['topic'].map({t: v['domain'] for t, v in TOP.items()})
def tc_(j): return re.sub(r'\s+', ' ', str(j).strip().lower())
enm['j'] = enm['journal'].apply(tc_)
core = {}
top5 = {d: [j for j, _ in __import__('collections').Counter(enm[enm.dom == d]['j']).most_common(5)] for d in DOMS}
for d in DOMS:
    core[d] = {'own': round(float(enm[enm.dom == d]['j'].isin(top5[d]).mean()), 4)}
    for d2 in DOMS:
        if d2 != d: core[d][d2] = round(float(enm[enm.dom == d2]['j'].isin(top5[d]).mean()), 3)
# document-weighted cross-domain venue sharing: share of a domain's articles appearing in journals that also carry the other domain
js = {d: set(enm[enm.dom == d]['j']) for d in DOMS}
docweighted = {f'{a}->{b}': round(float(enm[enm.dom == a]['j'].isin(js[b]).mean()), 3) for a in DOMS for b in DOMS if a != b}
json.dump(dict(core_top5=core, top5=top5, docweighted=docweighted), open(f'{OUT}/venue_core.json', 'w'), indent=1)
print('venue core:', core)

# ---------------------------------------------------------------- source-composition check and venue novelty
# (a) share trends 2005-2025 estimated separately within PubMed-matched and non-matched records: a trend that holds in both
#     parts of the corpus cannot be an artifact of the composition of one source; (b) share of 2015-2025 matched articles
#     published in journals that first appear in the corpus in 2013 or later (venue-expansion proxy), by domain.
en['matched'] = en['match'] != 'no-match'
en['dom'] = en['topic'].map({t: v['domain'] for t, v in TOP.items()})
def _slope(sub, mask):
    tot = sub.groupby('year').size().reindex(yrs, fill_value=0); cnt = sub[mask].groupby('year').size().reindex(yrs, fill_value=0)
    prop = (cnt / tot.replace(0, np.nan)).fillna(0).values * 100; r = stats.linregress(yrs, prop)
    return [round(float(r.slope * 10), 1), round(float(r.pvalue), 3)]
src = {'match_rate_2005_2014': round(float(en[en.year.between(2005, 2014)].matched.mean()), 3), 'match_rate_2015_2025': round(float(en[en.year.between(2015, 2025)].matched.mean()), 3),
       'n_matched_window': int(en[en.year.between(2005, 2025)].matched.sum()), 'n_unmatched_window': int((~en[en.year.between(2005, 2025)].matched).sum()), 'topics': {}, 'domains': {}}
m, u = en[en.matched], en[~en.matched]
for t in range(K): src['topics'][str(t)] = {'matched': _slope(m, m.topic == t), 'unmatched': _slope(u, u.topic == t)}
for d in DOMS: src['domains'][d] = {'matched': _slope(m, m.dom == d), 'unmatched': _slope(u, u.dom == d)}
# formal test of a source difference in trend: record-level binomial GLM, membership ~ year + source + year x source
# (2005-2025); the p-value of the interaction term is reported (unadjusted, one test per domain and for T0)
import statsmodels.api as sm
w = en[en.year.between(2005, 2025)].copy(); w['yc'] = (w['year'] - 2015) / 10.0; w['src'] = w['matched'].astype(float)
w['int'] = w['yc'] * w['src']
X = sm.add_constant(w[['yc', 'src', 'int']].values.astype(float))
src['interaction'] = {}
for name, mask in [(d, (w.dom == d)) for d in DOMS] + [('T0', (w.topic == 0))]:
    fit = sm.GLM(mask.values.astype(float), X, family=sm.families.Binomial()).fit()
    src['interaction'][name] = dict(b_year_unmatched=round(float(fit.params[1]), 3), b_interaction=round(float(fit.params[3]), 3), p_interaction=round(float(fit.pvalues[3]), 3))
enm2 = en[en.matched].copy(); enm2['j'] = enm2['journal'].astype(str).str.strip().str.lower(); enm2['jfirst'] = enm2['j'].map(enm2.groupby('j')['year'].min())
late = enm2[enm2.year.between(2015, 2025)]
venue_new = {'definition': 'share of 2015-2025 PubMed-matched articles in journals whose first article in the corpus is dated 2013 or later',
             'overall': round(float((late.jfirst >= 2013).mean()), 3), 'by_domain': {d: round(float((late[late.dom == d].jfirst >= 2013).mean()), 3) for d in DOMS},
             'top_new_journals': __import__('collections').Counter(enm2[enm2.jfirst >= 2013]['journal']).most_common(8)}
json.dump(dict(source_check=src, venue_new=venue_new), open(f'{OUT}/source_check.json', 'w'), indent=1)
print('source check:', src['domains'], venue_new['by_domain'])

# ---------------------------------------------------------------- representative documents per topic (15 closest to each centroid)
cent = np.load(f'{OUT}/centroids.npy'); En = E / np.linalg.norm(E, axis=1, keepdims=True); Cn = cent / np.linalg.norm(cent, axis=1, keepdims=True)
sim = En @ Cn.T; rows = []
for t in range(K):
    idx = np.where(df['topic'].values == t)[0]; order = idx[np.argsort(-sim[idx, t])][:15]
    for r, i in enumerate(order, 1):
        rows.append(dict(topic=t, label=TOP[t]['label'], domain=TOP[t]['domain'], rank=r, cosine=round(float(sim[i, t]), 3), year=int(df['Publication Year'].iloc[i]), title=df['Article Title'].iloc[i]))
pd.DataFrame(rows).to_csv(f'{OUT}/representative_documents.csv', index=False)
print('representative documents written:', len(rows))
