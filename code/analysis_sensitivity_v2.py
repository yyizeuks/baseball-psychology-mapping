"""Sensitivity and robustness analyses for the revised manuscript.

Imports analysis_pipeline_v2 and re-uses its corpus, representation and trend code.

Blocks (each writes out/sens_<block>.json):
  krange     K = 6..14: ARI with the primary solution, best-matching topic per primary
             topic with its trend slope / class, and domain-level trends
  krule      stability of the K-selection rule across sweep seeds
  svddim     SVD dimension 60..200: rule-selected K, ARI with primary at K=7
  bootstrap  Hennig-style bootstrap Jaccard stability per topic (100 resamples)
  lexicon    strict lexicon: retention, overlap, structure, trends
  trendmeth  OLS share / binomial GLM / CLR / Mann-Kendall / quadratic / era chi-square
  domains    hierarchical clustering of topic centroids + bootstrap co-grouping
  seeds      topic and domain trends under five alternative k-means initializations at the primary K
  substruct  sub-clustering of the largest topic (heterogeneity)
  emerging   emerging keywords (first appearance >= 2018)
"""
import json, re, sys, os
import numpy as np, pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize
import statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_pipeline_v2 as P

OUT = 'out'
SEED = 42
K0 = int(json.load(open(f'{OUT}/diagnostics.json'))['K']) if os.path.exists(f'{OUT}/diagnostics.json') else 7

def _domain_of_topic():
    """Domain of each primary topic: from labels.json when present, otherwise from the chosen hierarchical cut."""
    try:
        L = json.load(open('labels.json')); return {int(k): v['domain'] for k, v in L['topics'].items()}
    except Exception:
        d = json.load(open(f'{OUT}/sens_domains.json')); cut = d['cuts'][str(d['n_domains'])]
        return {t: 'ABCDE'[c - 1] for t, c in enumerate(cut)}
try:
    DOMAIN_OF = _domain_of_topic()
except Exception:
    DOMAIN_OF = {}

def load_primary():
    df = pd.read_csv(f'{OUT}/corpus_final.csv')
    E = np.load(f'{OUT}/E.npy'); cent = np.load(f'{OUT}/centroids.npy')
    docs = P.make_docs(df)
    _, Xc, vocab = P.represent(docs)
    docfreq = np.asarray((Xc > 0).sum(0)).ravel(); idf = np.log(1 + Xc.shape[0] / (docfreq + 1))
    Xbin = (Xc > 0).astype(np.int8).toarray()
    return df, E, cent, Xc, Xbin, docfreq, idf, vocab

def year_counts(df, lab, K, y0=2005, y1=2025):
    yrs = np.arange(y0, y1 + 1)
    sub = df.assign(topic=lab); sub = sub[sub['Publication Year'].isin(yrs)]
    tot = sub.groupby('Publication Year').size().reindex(yrs, fill_value=0).values
    cnt = np.vstack([sub[sub.topic == t].groupby('Publication Year').size().reindex(yrs, fill_value=0).values
                     for t in range(K)])
    return yrs, tot, cnt

def ols_share(yrs, tot, cnt):
    res = []
    for row in cnt:
        prop = np.where(tot > 0, row / np.maximum(tot, 1), 0) * 100
        sl, _, r, p, se = stats.linregress(yrs, prop)
        res.append((sl * 10, p))
    return res

def classify(slope, pfdr):
    return ('Hot' if slope > 0 and pfdr < .05 else 'Cold' if slope < 0 and pfdr < .05
            else 'positive (n.s.)' if slope > 0 else 'negative (n.s.)')

# ---------------------------------------------------------------- krange
def block_krange(df, E, cent, Xc, Xbin, docfreq, idf, vocab):
    lab0 = df['topic'].values
    yrs, tot, cnt0 = year_counts(df, lab0, K0)
    out = {'primary_K': K0, 'per_K': {}}
    for K in range(5, 15):  # K = 5 lies below the pre-specified range of the rule and is included for transparency
        lab, _, _, c, rows = P.fit_topics(E, Xc, Xbin, docfreq, idf, vocab, df, K, n_init=50)
        yrsK, totK, cnt = year_counts(df, lab, K)
        tr = ols_share(yrsK, totK, cnt)
        pf = P.bh_fdr([p for _, p in tr])
        sim = c @ cent.T  # K x K0
        matched = {}
        for t0 in range(K0):
            t = int(np.argmax(sim[:, t0]))
            purity = float(np.sum((lab == t) & (lab0 == t0)) / max(np.sum(lab == t), 1))
            matched[t0] = dict(matched_topic=t, cosine=round(float(sim[t, t0]), 3),
                               share_of_matched_from_primary=round(purity, 3),
                               b_per_decade=round(tr[t][0], 2), p_fdr=round(float(pf[t]), 3),
                               cls=classify(tr[t][0], pf[t]), n=int(np.sum(lab == t)),
                               keywords=rows[t]['keywords'].split('; ')[:6])
        out['per_K'][K] = dict(ari_vs_primary=round(adjusted_rand_score(lab0, lab), 3),
                               topics=matched,
                               all_topics=[dict(t=t, n=int(np.sum(lab == t)), b=round(tr[t][0], 2),
                                                p_fdr=round(float(pf[t]), 3), cls=classify(tr[t][0], pf[t]),
                                                kw=rows[t]['keywords'].split('; ')[:5]) for t in range(K)])
        # domain level: each alternative cluster is assigned the domain of its majority primary topic
        doms = sorted(set(DOMAIN_OF.values()))
        dom_of_alt = {}
        for t in range(K):
            counts = {d: int(np.sum((lab == t) & np.isin(lab0, [tt for tt in range(K0) if DOMAIN_OF[tt] == d]))) for d in doms}
            dom_of_alt[t] = max(doms, key=lambda d: counts[d])
        dl = np.array([doms.index(dom_of_alt[int(t)]) for t in lab])
        agree = float(np.mean([dom_of_alt[int(a)] == DOMAIN_OF[int(b)] for a, b in zip(lab, lab0)]))
        yrsD, totD, cntD = year_counts(df, dl, len(doms)); trd = ols_share(yrsD, totD, cntD); pfd = P.bh_fdr([p for _, p in trd])
        out['per_K'][K]['domain_agreement'] = round(agree, 3)
        out['per_K'][K]['domain_sizes'] = {d: int(np.sum(dl == i)) for i, d in enumerate(doms)}
        out['per_K'][K]['domain_trends'] = {d: [round(trd[i][0], 2), round(float(pfd[i]), 4)] for i, d in enumerate(doms)}
        print(f"K={K}: ARI vs primary {out['per_K'][K]['ari_vs_primary']}; domain agreement {agree:.3f} {out['per_K'][K]['domain_trends']}", flush=True)
    json.dump(out, open(f'{OUT}/sens_krange.json', 'w'), indent=1)

# ---------------------------------------------------------------- krule
def block_krule(df, E, Xc, Xbin, docfreq, idf):
    out = {}
    for seed in (1, 7, 13, 21):
        sweep = []
        for k in range(6, 19):
            lab = KMeans(k, n_init=20, random_state=seed).fit_predict(E)
            sil = float(silhouette_score(E, lab, random_state=SEED))
            coh = float(P.coherence_umass(Xbin, docfreq, P.ctfidf_order(Xc, lab, idf, k)).mean())
            sweep.append(dict(K=k, silhouette=round(sil, 4), umass=round(coh, 4), min_cluster=int(np.bincount(lab).min())))
        Ksel, rule = P.select_k(sweep)
        out[seed] = dict(K=Ksel, admissible=rule['admissible'], best_umass=rule['best_umass'], sweep=sweep)
        print(f"rule seed {seed}: K={Ksel} admissible={rule['admissible']}", flush=True)
    json.dump(out, open(f'{OUT}/sens_krule.json', 'w'), indent=1)

# ---------------------------------------------------------------- svddim
def block_svddim(df, Xc, Xbin, docfreq, idf):
    lab0 = df['topic'].values
    docs = P.make_docs(df)
    out = {}
    for d in (60, 90, 120, 150, 200):
        E, _, _ = P.represent(docs, n_components=d)
        sweep = P.k_sweep(E, Xc, Xbin, docfreq, idf, 6, 14)
        Ksel, rule = P.select_k(sweep)
        lab = KMeans(K0, n_init=50, random_state=SEED).fit_predict(E)
        # match the K0 refitted topics to the primary topics (Hungarian on the contingency table) and map documents to
        # the domains of their matched topics: domain-assignment agreement and domain-level trends at this dimension
        M = np.zeros((K0, K0))
        for a, b in zip(lab, lab0): M[a, b] += 1
        ri, ci = linear_sum_assignment(-M); mp = {int(a): int(b) for a, b in zip(ri, ci)}
        dom_alt = np.array([DOMAIN_OF.get(mp[int(l)], '?') for l in lab]); dom_prim = np.array([DOMAIN_OF.get(int(l), '?') for l in lab0])
        doms = sorted(set(DOMAIN_OF.values())); dl = np.array([doms.index(x) for x in dom_alt])
        yrs, tot, cnt = year_counts(df, dl, len(doms)); trd = ols_share(yrs, tot, cnt); pfd = P.bh_fdr([p for _, p in trd])
        # topic-level trends of the refitted topics, keyed by the primary topic each is matched to
        lab_m = np.array([mp[int(l)] for l in lab])
        yrsT, totT, cntT = year_counts(df, lab_m, K0); trt = ols_share(yrsT, totT, cntT); pft = P.bh_fdr([p for _, p in trt])
        out[d] = dict(rule_K=Ksel, admissible=rule['admissible'],
                      ari_vs_primary_at_K7=round(adjusted_rand_score(lab0, lab), 3),
                      silhouette_K7=round(float(silhouette_score(E, lab, random_state=SEED)), 4),
                      domain_agreement=round(float((dom_alt == dom_prim).mean()), 3),
                      domain_trends={dd: [round(trd[i][0], 2), round(float(pfd[i]), 4)] for i, dd in enumerate(doms)},
                      topic_trends={int(t): [round(trt[t][0], 2), round(float(pft[t]), 4), classify(trt[t][0], pft[t]),
                                             round(float(M[[a for a, b in mp.items() if b == t][0], t] / max(np.sum(lab_m == t), 1)), 3)]
                                    for t in range(K0)})
        print(f"dim {d}: rule K={Ksel}, ARI(K=7) vs primary {out[d]['ari_vs_primary_at_K7']}, domain agreement {out[d]['domain_agreement']}, {out[d]['domain_trends']}", flush=True)
    json.dump(out, open(f'{OUT}/sens_svddim.json', 'w'), indent=1)
    json.dump({d: dict(domain_agreement=v['domain_agreement'], domain_trends=v['domain_trends']) for d, v in out.items() if d != 120}, open(f'{OUT}/sens_svddim_domains.json', 'w'), indent=1)

# ---------------------------------------------------------------- bootstrap (Hennig 2007)
def block_bootstrap(df, E, cent, n_boot=100):
    lab0 = df['topic'].values
    rng = np.random.default_rng(SEED)
    jac = np.zeros((n_boot, K0))
    n = len(E)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        lab = KMeans(K0, n_init=10, random_state=b).fit_predict(E[idx])
        for t0 in range(K0):
            A = set(idx[lab0[idx] == t0])
            best = 0.0
            for t in range(K0):
                B = set(idx[lab == t])
                j = len(A & B) / max(len(A | B), 1)
                best = max(best, j)
            jac[b, t0] = best
        # domain-level stability: resampled clusters matched to primary topics (Hungarian), mapped to domains
        if DOMAIN_OF:
            doms = sorted(set(DOMAIN_OF.values()))
            M = np.zeros((K0, K0))
            for a, b0 in zip(lab, lab0[idx]): M[a, b0] += 1
            ri, ci = linear_sum_assignment(-M); mp = {int(a): int(b0) for a, b0 in zip(ri, ci)}
            dom_b = np.array([DOMAIN_OF[mp[int(l)]] for l in lab]); dom_0 = np.array([DOMAIN_OF[int(l)] for l in lab0[idx]])
            if b == 0: djac = np.zeros((n_boot, len(doms)))
            for i_, d in enumerate(doms):
                A = set(np.where(dom_0 == d)[0]); B = set(np.where(dom_b == d)[0])
                djac[b, i_] = len(A & B) / max(len(A | B), 1)
    out = dict(n_boot=n_boot, mean_jaccard=[round(float(x), 4) for x in jac.mean(0)],
               min_jaccard=[round(float(x), 4) for x in jac.min(0)],
               stable_ge_075=[bool(x >= 0.75) for x in jac.mean(0)],
               dissolved_frac=[round(float(x), 2) for x in (jac < 0.5).mean(0)],  # Hennig (2007): dissolved when Jaccard < 0.5
               overall_mean=round(float(jac.mean()), 3))
    if DOMAIN_OF:
        out['domain_jaccard'] = {d: round(float(djac[:, i_].mean()), 4) for i_, d in enumerate(doms)}
        out['domain_jaccard_min'] = {d: round(float(djac[:, i_].min()), 3) for i_, d in enumerate(doms)}
        out['domain_dissolved_frac'] = {d: round(float((djac[:, i_] < 0.5).mean()), 2) for i_, d in enumerate(doms)}
    print("bootstrap Jaccard per topic:", out['mean_jaccard'], flush=True)
    json.dump(out, open(f'{OUT}/sens_bootstrap.json', 'w'), indent=1)

# ---------------------------------------------------------------- lexicon (two stricter screens)
def _variant_trends(df, df_v, lab_prim_v, dom_of_prim):
    """Trends of the primary topics and domains within a variant corpus (documents keep their primary labels)."""
    yrs, tot, cnt = year_counts(df_v, lab_prim_v, K0); tr = ols_share(yrs, tot, cnt); pf = P.bh_fdr([p for _, p in tr])
    prim_t = {}
    for t in range(K0):
        n_t = int(np.sum(lab_prim_v == t))
        prim_t[t] = None if n_t < 15 else dict(n=n_t, b=round(tr[t][0], 2), p=float(tr[t][1]), p_fdr=round(float(pf[t]), 3), cls=classify(tr[t][0], pf[t]))
    dlab = np.array([dom_of_prim[int(t)] for t in lab_prim_v]); doms = sorted(set(dom_of_prim.values()))
    dl = np.array([doms.index(d) for d in dlab])
    yrs, tot, cnt = year_counts(df_v, dl, len(doms)); trd = ols_share(yrs, tot, cnt); pfd = P.bh_fdr([p for _, p in trd])
    dom_t = {d: [round(trd[i][0], 2), round(float(pfd[i]), 3)] for i, d in enumerate(doms)}
    return prim_t, dom_t

def _lexicon_variant(df, lexicon, name, fname):
    df_s, cascade_s, _, _ = P.build_corpus('corpus_raw.xlsx', lexicon=lexicon)
    key = lambda s: re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', str(s).lower())).strip()
    prim = {key(t): i for i, t in enumerate(df['Article Title'])}
    in_prim = np.array([key(t) in prim for t in df_s['Article Title']])
    docs = P.make_docs(df_s)
    E, Xc, vocab = P.represent(docs)
    docfreq = np.asarray((Xc > 0).sum(0)).ravel(); idf = np.log(1 + Xc.shape[0] / (docfreq + 1))
    Xbin = (Xc > 0).astype(np.int8).toarray()
    sweep = P.k_sweep(E, Xc, Xbin, docfreq, idf, 6, 12)
    mc = max(12, int(round(25 * len(df_s) / len(df))))
    Ksel, rule = P.select_k(sweep, min_cluster=mc)
    lab, _, _, c, rows = P.fit_topics(E, Xc, Xbin, docfreq, idf, vocab, df_s, K0, n_init=20)
    yrs, tot, cnt = year_counts(df_s, lab, K0); tr = ols_share(yrs, tot, cnt); pf = P.bh_fdr([p for _, p in tr])
    lab0_shared = np.array([df.loc[prim[key(t)], 'topic'] for t, s_ in zip(df_s['Article Title'], in_prim) if s_])
    lab_shared = lab[in_prim]
    M = np.zeros((K0, K0))
    for a, b in zip(lab_shared, lab0_shared): M[a, b] += 1
    ri, ci = linear_sum_assignment(-M)
    mapping = {int(a): int(b) for a, b in zip(ri, ci)}
    # match quality: share of the refitted topic's shared documents that carry the matched primary label
    quality = {int(a): round(float(M[a, mapping[a]] / max(M[a].sum(), 1)), 3) for a in range(K0)}
    # primary-label trends within the variant corpus
    lab_prim_v = np.array([df.loc[prim[key(t)], 'topic'] if s_ else -1 for t, s_ in zip(df_s['Article Title'], in_prim)])
    dfv = df_s[in_prim].reset_index(drop=True); lab_prim_v = lab_prim_v[in_prim]
    dom_of_prim = {int(t): DOMAIN_OF[t] for t in range(K0)}
    prim_t, dom_t = _variant_trends(df, dfv, lab_prim_v, dom_of_prim)
    out = dict(name=name, cascade=cascade_s, n=int(len(df_s)), n_strict=int(len(df_s)), n_in_primary=int(in_prim.sum()),
               share_of_primary_retained=round(float(in_prim.sum() / len(df)), 3),
               rule_K=Ksel, admissible=rule['admissible'], min_cluster=mc,
               ari_shared=round(adjusted_rand_score(lab0_shared, lab_shared), 3), ari_shared_docs=round(adjusted_rand_score(lab0_shared, lab_shared), 3),
               topics=[dict(strict_t=t, primary_match=mapping[t], match_quality=quality[t], n=int(np.sum(lab == t)),
                            b=round(tr[t][0], 2), p_fdr=round(float(pf[t]), 3), cls=classify(tr[t][0], pf[t]),
                            kw=rows[t]['keywords'].split('; ')[:6]) for t in range(K0)],
               primary_topic_trends_in_variant={str(t): v for t, v in prim_t.items()}, domain_trends=dom_t)
    print(f"{name}: n={len(df_s)} retention={cascade_s['retention_rate']} K={Ksel} ARI(shared)={out['ari_shared']} dom={dom_t}", flush=True)
    json.dump(out, open(f'{OUT}/{fname}', 'w'), indent=1)

def block_lexicon(df, cent):
    _lexicon_variant(df, P.PSYCH_LEXICON_STRICT, 'strict lexicon', 'sens_lexicon.json')
    lex_nostress = '|'.join(e for e in P.lexicon_entries(P.PSYCH_LEXICON_STRICT) if e != 'stress')
    _lexicon_variant(df, lex_nostress, "strict lexicon without 'stress'", 'sens_lexicon2.json')

# ---------------------------------------------------------------- trend methods
def mann_kendall(x):
    n = len(x); s = 0
    for i in range(n - 1):
        s += np.sum(np.sign(x[i + 1:] - x[i]))
    var = n * (n - 1) * (2 * n + 5) / 18
    z = (s - 1) / np.sqrt(var) if s > 0 else (s + 1) / np.sqrt(var) if s < 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    slopes = [(x[j] - x[i]) / (j - i) for i in range(n) for j in range(i + 1, n)]
    return float(s), float(z), float(p), float(np.median(slopes))

def block_trendmeth(df):
    lab0 = df['topic'].values
    yrs, tot, cnt = year_counts(df, lab0, K0)
    res = {}
    ols = ols_share(yrs, tot, cnt); p_ols = P.bh_fdr([p for _, p in ols])
    glm, clr, mk, quad = [], [], [], []
    X = sm.add_constant((yrs - yrs.mean()) / 10.0)  # decade units, centered
    comp = (cnt + 0.5) / (tot + 0.5 * K0)
    logg = np.log(comp)
    clr_mat = logg - logg.mean(0, keepdims=True)
    for t in range(K0):
        # binomial GLM on counts with yearly denominators
        y = np.column_stack([cnt[t], tot - cnt[t]])
        m = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        glm.append((float(m.params[1]), float(m.pvalues[1])))
        # CLR-OLS
        sl, _, r, p, se = stats.linregress(yrs, clr_mat[t]); clr.append((sl * 10, p))
        # Mann-Kendall on shares
        prop = cnt[t] / np.maximum(tot, 1) * 100
        s, z, p, sen = mann_kendall(prop); mk.append((sen * 10, p))
        # quadratic term F-test
        X2 = np.column_stack([np.ones_like(yrs), yrs - 2015, (yrs - 2015) ** 2])
        m1 = sm.OLS(prop, X2[:, :2]).fit(); m2 = sm.OLS(prop, X2).fit()
        fp = float(m2.compare_f_test(m1)[1]); quad.append(fp)
    p_glm = P.bh_fdr([p for _, p in glm]); p_clr = P.bh_fdr([p for _, p in clr]); p_mk = P.bh_fdr([p for _, p in mk]); quad_fdr = P.bh_fdr(quad)
    for t in range(K0):
        res[t] = dict(ols_b=round(ols[t][0], 2), ols_pfdr=round(float(p_ols[t]), 6), ols_cls=classify(ols[t][0], p_ols[t]),
                      glm_logit_b_per_decade=round(glm[t][0], 3), glm_pfdr=round(float(p_glm[t]), 6), glm_cls=classify(glm[t][0], p_glm[t]),
                      clr_b=round(clr[t][0], 3), clr_pfdr=round(float(p_clr[t]), 6), clr_cls=classify(clr[t][0], p_clr[t]),
                      mk_sen_per_decade=round(mk[t][0], 2), mk_pfdr=round(float(p_mk[t]), 6), mk_cls=classify(mk[t][0], p_mk[t]),
                      quadratic_p=round(quad[t], 6), quadratic_pfdr=round(float(quad_fdr[t]), 6))
    # era-level chi-square
    eras = [(1990, 1999), (2000, 2009), (2010, 2014), (2015, 2019), (2020, 2026)]
    tab = np.array([[int(np.sum((df['Publication Year'].between(a, b)) & (lab0 == t))) for t in range(K0)] for a, b in eras])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    era_share = (tab / tab.sum(1, keepdims=True) * 100).round(1).tolist()
    out = dict(per_topic=res, era_table=tab.tolist(), era_share_pct=era_share, eras=[f"{a}-{b}" for a, b in eras],
               era_chi2=round(float(chi2), 2), era_p=float(p), era_dof=int(dof),
               cramers_v=round(float(np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1)))), 3),
               n_years=int(len(yrs)), yearly_totals=tot.tolist())
    json.dump(out, open(f'{OUT}/sens_trendmeth.json', 'w'), indent=1)
    for t, r in res.items(): print(t, r['ols_cls'], r['glm_cls'], r['clr_cls'], r['mk_cls'], 'quad p', r['quadratic_p'])

# ---------------------------------------------------------------- domains
def block_domains(df, E, cent, n_boot=200):
    D = pdist(cent, metric='cosine'); Z = linkage(D, method='average')
    coph = float(cophenet(Z, D)[0])
    cuts = {k: fcluster(Z, k, criterion='maxclust').tolist() for k in (2, 3, 4, 5)}
    lab0 = df['topic'].values
    # silhouette of document-level domain assignment for each cut; the number of domains is the cut that maximizes it
    sil = {}
    for k, cut in cuts.items():
        dlab = np.array([cut[t] for t in lab0])
        sil[k] = round(float(silhouette_score(E, dlab, random_state=SEED)), 4)
    k_dom = int(max(sil, key=lambda k: sil[k]))
    # bootstrap co-grouping: documents resampled with replacement, topic labels held fixed, centroids recomputed,
    # hierarchical clustering re-cut at the chosen number of domains (and, for reference, at 4)
    rng = np.random.default_rng(SEED)
    co = {k: np.zeros((K0, K0)) for k in (k_dom, 4)}; n = len(E)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        cb = normalize(np.vstack([E[idx][lab0[idx] == t].mean(0) if np.any(lab0[idx] == t) else cent[t] for t in range(K0)]))
        Zb = linkage(pdist(cb, 'cosine'), 'average')
        for k in co:
            fb = fcluster(Zb, k, criterion='maxclust'); co[k] += (fb[:, None] == fb[None, :])
    for k in co: co[k] /= n_boot
    out = dict(linkage=Z.tolist(), cophenetic=round(coph, 3), cuts=cuts, doc_silhouette_by_cut=sil, n_domains=k_dom,
               bootstrap_k=k_dom, bootstrap_cogrouping=co[k_dom].round(3).tolist(), bootstrap_cogrouping_4=co[4].round(3).tolist(),
               bootstrap_note='topic labels held fixed; documents resampled with replacement; centroids recomputed; tree re-cut',
               centroid_cosine=(1 - squareform(D)).round(3).tolist())
    json.dump(out, open(f'{OUT}/sens_domains.json', 'w'), indent=1)
    print("domain cuts:", cuts, "| cophenetic", round(coph, 3), "| doc-sil by cut", sil)

# ---------------------------------------------------------------- seeds (topic and domain trends across initializations)
def block_seeds(df, E):
    lab0 = df['topic'].values
    out = {}
    doms = sorted(set(DOMAIN_OF.values()))
    for seed in (1, 7, 13, 21, 99):
        lab = KMeans(K0, n_init=50, random_state=seed).fit_predict(E)
        M = np.zeros((K0, K0))
        for a, b in zip(lab, lab0): M[a, b] += 1
        ri, ci = linear_sum_assignment(-M); mp = {int(a): int(b) for a, b in zip(ri, ci)}
        lab_m = np.array([mp[int(l)] for l in lab])
        yrsT, totT, cntT = year_counts(df, lab_m, K0); trt = ols_share(yrsT, totT, cntT); pft = P.bh_fdr([p for _, p in trt])
        dl = np.array([doms.index(DOMAIN_OF[int(t)]) for t in lab_m])
        yrs, tot, cnt = year_counts(df, dl, len(doms)); trd = ols_share(yrs, tot, cnt); pfd = P.bh_fdr([p for _, p in trd])
        out[seed] = dict(ari_vs_primary=round(adjusted_rand_score(lab0, lab), 3),
                         domain_agreement=round(float(np.mean([DOMAIN_OF[int(a)] == DOMAIN_OF[int(b)] for a, b in zip(lab_m, lab0)])), 3),
                         topic_trends={int(t): [round(trt[t][0], 2), round(float(pft[t]), 4), classify(trt[t][0], pft[t])] for t in range(K0)},
                         domain_trends={dd: [round(trd[i][0], 2), round(float(pfd[i]), 4)] for i, dd in enumerate(doms)})
        print(f"seed {seed}: ARI {out[seed]['ari_vs_primary']} domain agreement {out[seed]['domain_agreement']} {out[seed]['domain_trends']}", flush=True)
    json.dump(out, open(f'{OUT}/sens_seeds.json', 'w'), indent=1)

# ---------------------------------------------------------------- substructure
def block_substruct(df, E, Xc, docfreq, idf, vocab):
    lab0 = df['topic'].values
    sizes = np.bincount(lab0); big = [int(t) for t in np.argsort(-sizes)[:2]]
    # plus the least coherent topic and any topic that forms a domain on its own
    t1 = pd.read_csv(f'{OUT}/table1_topics.csv')
    least = int(t1.sort_values('coherence').iloc[0]['topic_id'])
    if least not in big: big.append(least)
    if DOMAIN_OF:
        from collections import Counter
        cnt_d = Counter(DOMAIN_OF.values())
        for t, dd in DOMAIN_OF.items():
            if cnt_d[dd] == 1 and t not in big: big.append(int(t))
    out = {}
    for t in big:
        m = lab0 == t
        res = {}
        for k in (2, 3):
            sub = KMeans(k, n_init=20, random_state=SEED).fit_predict(E[m])
            full = np.full(len(lab0), -1); full[m] = sub
            orders = [np.argsort((np.asarray(Xc[full == s].sum(0)).ravel() / max((full == s).sum(), 1)) * idf)[::-1][:8] for s in range(k)]
            res[k] = dict(sil=round(float(silhouette_score(E[m], sub, random_state=SEED)), 3),
                          sub=[dict(n=int(np.sum(sub == s)), kw=[str(w) for w in vocab[orders[s]]]) for s in range(k)])
        out[t] = dict(n=int(m.sum()), splits=res)
        print(f"topic {t} (n={m.sum()}):", {k: [x['kw'][:4] for x in v['sub']] for k, v in res.items()})
    json.dump(out, open(f'{OUT}/sens_substruct.json', 'w'), indent=1)

# ---------------------------------------------------------------- emerging keywords
def block_emerging(df):
    kw = df[['Publication Year', 'Keywords']].dropna()
    rows = []
    for y, k in zip(kw['Publication Year'], kw['Keywords']):
        for term in re.split(r'[;,|]', str(k)):
            term = term.strip().lower()
            if 2 < len(term) < 60: rows.append((int(y), term))
    t = pd.DataFrame(rows, columns=['year', 'term'])
    g = t.groupby('term')
    stat = pd.DataFrame(dict(n=g.size(), first=g['year'].min(), recent=g.apply(lambda s: int((s['year'] >= 2023).sum()))))
    stat = stat[stat.n >= 5]
    emerging = stat[stat['first'] >= 2018].sort_values('n', ascending=False).head(25)
    stat['recent_share'] = stat['recent'] / stat['n']
    rising = stat[stat.n >= 8].sort_values('recent_share', ascending=False).head(25)
    out = dict(n_records_with_keywords=int(len(kw)), share_with_keywords=round(float(len(kw) / len(df)), 3),
               emerging_first2018=emerging.reset_index().to_dict('records'),
               rising_recent_share=rising.reset_index().to_dict('records'))
    json.dump(out, open(f'{OUT}/sens_emerging.json', 'w'), indent=1, default=int)
    print("emerging:", emerging.index.tolist()[:12])

if __name__ == '__main__':
    blocks = sys.argv[1:] or ['krange', 'krule', 'svddim', 'bootstrap', 'lexicon', 'trendmeth', 'domains', 'seeds', 'substruct', 'emerging']
    df, E, cent, Xc, Xbin, docfreq, idf, vocab = load_primary()
    for b in blocks:
        print(f"\n=== {b} ===", flush=True)
        if b == 'krange': block_krange(df, E, cent, Xc, Xbin, docfreq, idf, vocab)
        elif b == 'krule': block_krule(df, E, Xc, Xbin, docfreq, idf)
        elif b == 'svddim': block_svddim(df, Xc, Xbin, docfreq, idf)
        elif b == 'bootstrap': block_bootstrap(df, E, cent)
        elif b == 'lexicon': block_lexicon(df, cent)
        elif b == 'trendmeth': block_trendmeth(df)
        elif b == 'domains': block_domains(df, E, cent)
        elif b == 'seeds': block_seeds(df, E)
        elif b == 'substruct': block_substruct(df, E, Xc, docfreq, idf, vocab)
        elif b == 'emerging': block_emerging(df)
