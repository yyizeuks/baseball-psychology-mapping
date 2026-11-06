"""
Baseball psychology knowledge structure - reproducible analysis pipeline (v2, revision)
======================================================================================

Reproduces every number reported in the revised manuscript

  Lee, J. W., & Han, D. H. Mapping the Knowledge Structure of Baseball
  Psychology Research: A Systematic Mapping Review Using Topic Modeling
  and an Auditable Relevance Screen. (Frontiers in Psychology, revision 1)

Changes from v1 (submitted version)
  * De-duplication now uses a normalized title key (lower-case, punctuation and
    whitespace collapsed) with an abstract-similarity / year check inside each
    key group, instead of exact lower-case title matching. v1 missed
    cross-database duplicates that differed only in punctuation or casing.
  * The screening cascade records every exclusion count, including the number
    of records removed by the language heuristic and the abstract-completeness
    exclusions broken down by publication period.
  * Model selection applies an explicit numeric rule (see select_k) instead of a
    narrative choice.
  * Sensitivity analyses (lexicon strictness, K, SVD dimension, random seeds,
    compositional / GLM / non-parametric trend tests) live in
    analysis_sensitivity_v2.py and import this module.

Requirements
    pip install pandas numpy scipy scikit-learn openpyxl

Run
    python analysis_pipeline_v2.py --input baseball_psychology_final.xlsx

Outputs (./out/)
    cascade.json           screening cascade with all exclusion counts
    duplicates.csv         every duplicate group that was collapsed, with the kept record
    table1_topics.csv      topic keywords, sizes, coherence, representative docs
    table2_trends.csv      OLS on annual topic share, BH-FDR adjusted p-values
    diagnostics.json       K sweep, silhouette, seed ARI, leave-p-out, temporal subsets
    doc_assignments.csv    per-article topic assignment (title, year, topic)
    corpus_final.csv       the screened corpus (all fields) for deposit
    relevance_lexicon.txt  both screening lexicons, verbatim

All randomness is seeded (SEED = 42); reruns are identical.
"""

import argparse
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer,
                                             ENGLISH_STOP_WORDS)
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

SEED = 42
OUT = "out"

# ==========================================================================
# Stage 1 - corpus construction (Section 2.1)
# ==========================================================================

PSYCH_LEXICON = r"""psycholog|mental|cognit|percept|attention|anticipat|decision|motivation|
emotion|anxiety|depress|stress|burnout|confidence|self-efficacy|efficacy|arousal|mood|
wellbeing|well-being|personality|behavio|social|team|leader|coach|fan|spectator|identity|
identit|fan identification|team identification|motor learning|motor control|skill acquisi|expertise|practice|learning|memory|
attitude|adhd|ptsd|concussion|neurocognit|brain|neural|eeg|fmri|gaze|eye movement|
visual search|reaction time|quiet eye|imagery|mindfulness|coping|resilien|adherence|
fatigue|sleep|substance|alcohol|tobacco|doping|quality of life|satisfaction|engagement|
dropout|youth development|parent|student-athlete|retirement|career|umpire|judgment|bias|
expectancy|choking|pressure|self-talk|goal setting|group dynamic"""

BASEBALL_LEXICON = r"""baseball|softball|pitcher|pitchers|pitching|\bbatter(?![yi])|batting|hitter|hitters|
ballplayer|ballplayers|ballpark|mlb|major league|minor league|little league|npb|kbo"""
# One mention = one baseball-specific phrase. Compound phrases are listed first so that a league name combined with the
# sport name and its acronym ('Major League Baseball (MLB)', 'Little League baseball') and a sport name followed by a
# role term ('baseball pitchers') each count once; 'baseball cap(s)' is not a mention.
BASEBALL_MENTION = (
    r"\b(?:"
    r"(?:major|minor|little)\s+league\s+(?:baseball|softball)\b(?:\s*\(\s*(?:mlb|milb|llb)\s*\))?"
    r"|(?:baseball|softball)\s+(?:pitchers?|pitching|batters?|batting|hitters?|ballplayers?|players?)\b"
    r"|(?:baseball|softball)\b(?!\s+caps?\b)"
    r"|(?:pitchers?|pitching|batter(?![yi])s?|batting|hitters?|ballplayers?|ballpark|mlb)\b"
    r"|(?:major\s+league|minor\s+league|little\s+league|npb|kbo)\b"
    r")")
BASEBALL_SPORT = r"\b(?:baseball|softball)\b"  # kept for reference


def baseball_mentions(field):
    """Number of baseball-specific mentions in each string (see BASEBALL_MENTION)."""
    return field.str.count(BASEBALL_MENTION)

PROCEDURAL_LEXICON = r"""arthroscop|surgical technique|surgery|reconstruct|fixation|graft|
suture|radiograph|mri|ultrasound|imaging|histolog|cadaver|anatomic|biopsy|osteochondr|
dissecans|capitell|hamate|olecranon|spondylol|ligament|tendon|lesion|fracture"""

# Stricter lexicon used in the sensitivity analysis: biomechanics-adjacent and
# generic social terms removed, so that only explicitly psychological content is retained.
PSYCH_LEXICON_STRICT = r"""psycholog|mental|cognit|percept|attention|anticipat|decision|motivation|
emotion|anxiety|depress|stress|burnout|confidence|self-efficacy|arousal|mood|
wellbeing|well-being|personality|motor learning|skill acquisi|expertise|memory|
attitude|adhd|ptsd|neurocognit|brain|neural|eeg|fmri|gaze|eye movement|
visual search|reaction time|quiet eye|imagery|mindfulness|coping|resilien|
quality of life|satisfaction|dropout|choking|self-talk|goal setting|group dynamic"""

DOMAIN_STOP = """baseball study studies result results conclusion conclusions purpose method
methods significant significantly analysis data using used based aim objective background
pubmed copyright abstract article research paper elsevier rights reserved p lt gt group
groups compared associated may also however findings finding present examined investigated
des la le der und mlb league major participants subjects level levels effect effects year
years age old sport sports athlete athletes player players game games play playing season
increased decreased higher lower greater time included measured assessed evaluated
respectively total mean sd""".split()

PERIODS = [(1914, 1989, "pre-1990"), (1990, 1999, "1990-1999"), (2000, 2009, "2000-2009"),
           (2010, 2019, "2010-2019"), (2020, 2026, "2020-2026")]


# Lexicon entries are matched as word-initial stems (\b + stem, so that "mental" matches "mental" and
# "mentally" but not "experimental"), multi-word entries are matched as phrases, and five entries carry
# context exclusions ("confidence" not followed by "interval"; "learning" not preceded by "machine";
# "pressure" not preceded by "blood"; "bias" not preceded by selection/publication/recall/reporting).
LEXICON_CONTEXT = {
    "confidence": r"\bconfidence(?!\s+interval)",
    "learning": r"(?<!machine )\blearning",
    "pressure": r"(?<!blood )\bpressure",
    "bias": r"(?<!selection )(?<!publication )(?<!recall )(?<!reporting )\bbias",
    "fan": r"\bfan(?!ning\b)(?!s?\s+(?:cool|speed|blade|motor))",
}


def lexicon_entries(lex):
    """Return the lexicon as a list of entries (stems and phrases) in the order written."""
    return [re.sub(r"\s+", " ", e.strip()) for e in re.sub(r"\s*\n\s*", "", lex).split("|") if e.strip()]


def compile_lexicon(lex):
    """Build one alternation pattern from a lexicon: word-initial stems, phrases with flexible spaces."""
    pats = []
    for e in lexicon_entries(lex):
        if e in LEXICON_CONTEXT:
            pats.append(LEXICON_CONTEXT[e])
        elif e.startswith("\\b"):
            pats.append(e)
        else:
            pats.append(r"\b" + e.replace(" ", r"\s+"))
    return "(?:" + "|".join(pats) + ")"


def count_matches(field, lex):
    """Number of lexicon matches in each string of a pandas Series (lower-cased text)."""
    return field.str.count(compile_lexicon(lex))


def norm_title(s):
    """Lower-case, strip diacritics (so that 'Sueños de Béisbol' and 'Suenos de Beisbol' share a key), drop punctuation."""
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def word_jaccard(a, b):
    A = set(re.findall(r"[a-z]{4,}", str(a).lower()))
    B = set(re.findall(r"[a-z]{4,}", str(b).lower()))
    return len(A & B) / max(len(A | B), 1)


def deduplicate(df):
    """Collapse cross-database duplicates.

    Key: normalized title. Inside a key group two records are the same work when
    (a) their abstracts share >= 0.5 word Jaccard, or (b) either abstract is
    missing, or (c) their publication years differ by at most one. The retained
    record is the most complete one (abstract present, longest abstract, keywords
    present). Records with an empty title are keyed on the normalized abstract.
    """
    df = df.copy()
    df["_key"] = df["Article Title"].map(norm_title)
    empty = df["_key"] == ""
    df.loc[empty, "_key"] = "ABS::" + df.loc[empty, "Abstract"].map(norm_title)
    df["_abs_len"] = df["Abstract"].fillna("").astype(str).str.len()
    df["_kw"] = df["Keywords"].notna().astype(int)

    keep, log = [], []
    for key, g in df.groupby("_key", sort=False):
        if len(g) == 1:
            keep.append(g.index[0]); continue
        g = g.sort_values(["_abs_len", "_kw"], ascending=False)
        clusters = []  # list of lists of indices
        for idx, row in g.iterrows():
            placed = False
            for cl in clusters:
                ref = df.loc[cl[0]]
                a, b = ref["Abstract"], row["Abstract"]
                same = (pd.isna(a) or pd.isna(b) or word_jaccard(a, b) >= 0.5
                        or abs(int(ref["Publication Year"]) - int(row["Publication Year"])) <= 1)
                if same:
                    cl.append(idx); placed = True; break
            if not placed:
                clusters.append([idx])
        for cl in clusters:
            keep.append(cl[0])
            if len(cl) > 1:
                log.append(dict(key=key[:120], n_records=len(cl),
                                years="|".join(str(int(df.loc[i, "Publication Year"])) for i in cl),
                                kept_year=int(df.loc[cl[0], "Publication Year"])))
    out = df.loc[sorted(keep)].drop(columns=["_key", "_abs_len", "_kw"])
    return out, pd.DataFrame(log)


def is_english(x):
    """Character-composition heuristic: share of ASCII letters among all characters (CJK letters do not count)."""
    x = str(x)
    return len(re.findall(r"[A-Za-z]", x)) / max(len(x), 1) > 0.55


NON_RESEARCH_TITLE = (r"\(\s*1[89]\d{2}\s*[\u2013\u2014-]\s*(?:19|20)\d{2}\s*\)\.?\s*$|\bobituar|\bin memoriam\b|\binterview by\b|"
                      r"\berratum\b|\bcorrigend|^correction to\b|^retraction")


def load_non_research_list():
    """Normalized titles of records whose PubMed publication type is Editorial, Letter, News, Interview, Biography or
    Portrait (data/non_research_records.csv, produced from the PubMed enrichment); empty list if the file is absent."""
    for p in ("non_research_records.csv", "data/non_research_records.csv", "../data/non_research_records.csv"):
        if os.path.exists(p):
            t = pd.read_csv(p)
            return set(t["Article Title"].map(norm_title))
    return set()


def build_corpus(path, sheet="baseball_psychology", lexicon=PSYCH_LEXICON, min_psych=2):
    """Retrieval -> de-duplication -> publication type -> completeness -> language -> window -> topicality -> relevance."""
    df = pd.read_excel(path, sheet_name=sheet)
    cascade = {"retrieved": len(df)}

    df, dup_log = deduplicate(df)
    cascade["duplicates_removed"] = cascade["retrieved"] - len(df)
    cascade["after_deduplication"] = len(df)

    # publication type: obituaries, biographies, interviews, errata, editorials and letters are not research articles.
    # Two sources: title patterns, and PubMed publication types for records matched to PubMed (deposited list).
    nonres = df["Article Title"].fillna("").str.contains(NON_RESEARCH_TITLE, regex=True, case=False)
    listed = load_non_research_list()
    if listed:
        nonres = nonres | df["Article Title"].map(norm_title).isin(listed)
    cascade["excluded_publication_type"] = int(nonres.sum())
    cascade["excluded_publication_type_titles"] = df.loc[nonres, "Article Title"].tolist()
    df = df[~nonres]
    cascade["after_publication_type"] = len(df)

    ok_abs = df["Abstract"].notna() & (df["Abstract"].astype(str).str.len() >= 200)
    excl = df[~ok_abs]
    cascade["excluded_abstract_total"] = int((~ok_abs).sum())
    cascade["excluded_abstract_missing"] = int(df["Abstract"].isna().sum())
    cascade["excluded_abstract_short"] = int((df["Abstract"].notna() & ~ok_abs).sum())
    cascade["excluded_abstract_by_period"] = {
        name: int(((excl["Publication Year"] >= a) & (excl["Publication Year"] <= b)).sum())
        for a, b, name in PERIODS}
    cascade["records_by_period_after_dedup"] = {
        name: int(((df["Publication Year"] >= a) & (df["Publication Year"] <= b)).sum())
        for a, b, name in PERIODS}
    df = df[ok_abs]
    cascade["after_abstract_completeness"] = len(df)

    eng = df["Abstract"].apply(is_english)
    cascade["excluded_language"] = int((~eng).sum())
    df = df[eng]
    cascade["after_language"] = len(df)

    win = (df["Publication Year"] >= 1990) & (df["Publication Year"] <= 2026)
    cascade["excluded_year_window"] = int((~win).sum())
    df = df[win]
    cascade["after_year_window"] = len(df)

    # topicality: baseball must be the object of study (a baseball-specific term in the title, or at least two
    # such terms in the abstract and keywords combined)
    n_bb_title = baseball_mentions(df["Article Title"].fillna("").str.lower())
    n_bb_body = baseball_mentions((df["Abstract"].fillna("") + " " + df["Keywords"].fillna("")).str.lower())
    topical = (n_bb_title >= 1) | (n_bb_body >= 2)
    cascade["excluded_topicality"] = int((~topical).sum())
    df = df.assign(_n_bb_title=n_bb_title.values, _n_bb_body=n_bb_body.values)
    topical_out = df[~topical].reset_index(drop=True)
    df = df[topical]
    cascade["after_topicality"] = len(df)

    field = (df["Article Title"].fillna("") + " " + df["Abstract"].fillna("") + " "
             + df["Keywords"].fillna("")).str.lower()
    n_psych, n_proc = count_matches(field, lexicon), count_matches(field, PROCEDURAL_LEXICON)
    keep = (n_psych >= min_psych) & (n_psych >= n_proc)
    cascade["excluded_relevance"] = int((~keep).sum())
    df = df.assign(_n_psych=n_psych.values, _n_proc=n_proc.values)
    screened_out = df[~keep].reset_index(drop=True)
    df = df[keep]
    cascade["after_relevance_screen"] = len(df)
    cascade["retention_rate"] = round(
        cascade["after_relevance_screen"] / cascade["after_topicality"], 4)
    cascade["retention_rate_of_eligible"] = round(
        cascade["after_relevance_screen"] / cascade["after_year_window"], 4)
    screened_out = pd.concat([topical_out.assign(_excluded_by="topicality"),
                              screened_out.assign(_excluded_by="relevance")], ignore_index=True)

    return df.reset_index(drop=True), cascade, dup_log, screened_out


# ==========================================================================
# Stage 2 - latent-semantic document representation (Section 2.2)
# ==========================================================================

def make_docs(df):
    return (df["Article Title"].fillna("") + ". " + df["Abstract"].fillna("") + " "
            + df["Keywords"].fillna("")).tolist()


def represent(docs, n_components=120):
    """TF-IDF -> truncated SVD -> unit norm. Also returns the count matrix."""
    stop = list(ENGLISH_STOP_WORDS.union(DOMAIN_STOP))
    clean = lambda x: re.sub(r"[^a-z\s\-]", " ", x.lower())
    dc = [clean(d) for d in docs]

    cv = CountVectorizer(stop_words=stop, min_df=8, max_df=0.4,
                         token_pattern=r"[a-z][a-z\-]{2,}")
    Xc = cv.fit_transform(dc)

    tv = TfidfVectorizer(stop_words=stop, min_df=8, max_df=0.4,
                         token_pattern=r"[a-z][a-z\-]{2,}", sublinear_tf=True)
    E = normalize(TruncatedSVD(n_components, random_state=SEED)
                  .fit_transform(tv.fit_transform(dc)))
    return E, Xc, np.array(cv.get_feature_names_out())


# ==========================================================================
# Stage 3 - topic induction, model selection, diagnostics (Section 2.3)
# ==========================================================================

def ctfidf_order(Xc, labels, idf, K):
    """Class-based TF-IDF: term indices ordered by discriminative weight per topic."""
    return [np.argsort((np.asarray(Xc[labels == t].sum(0)).ravel()
                        / max((labels == t).sum(), 1)) * idf)[::-1] for t in range(K)]


def coherence_umass(Xbin, docfreq, orders, topn=10):
    """C_UMass over the binary document-term matrix. Values near zero are better."""
    out = []
    for idx in orders:
        idx = idx[:topn]
        total = pairs = 0
        for i in range(1, len(idx)):
            for j in range(i):
                co = int(np.sum(Xbin[:, idx[i]] & Xbin[:, idx[j]]))
                total += np.log((co + 1) / docfreq[idx[j]])
                pairs += 1
        out.append(total / pairs)
    return np.array(out)


def k_sweep(E, Xc, Xbin, docfreq, idf, kmin, kmax):
    sweep = []
    for k in range(kmin, kmax + 1):
        lab = KMeans(k, n_init=50, random_state=SEED).fit_predict(E)  # same settings as the final fit
        sil = float(silhouette_score(E, lab, random_state=SEED))
        coh = float(coherence_umass(Xbin, docfreq, ctfidf_order(Xc, lab, idf, k)).mean())
        sweep.append(dict(K=k, silhouette=round(sil, 4), umass=round(coh, 4),
                          min_cluster=int(np.bincount(lab).min())))
    return sweep


def select_k(sweep, min_cluster=25):
    """Explicit selection rule.

    Admissible K: smallest topic >= min_cluster documents (topics smaller than
    this cannot be characterized or trended reliably). Among admissible K, the
    selected solution is the one with the highest mean C_UMass coherence; ties
    are broken by the silhouette coefficient. Stability across initializations,
    resamples and neighboring K is reported as a diagnostic of the selected
    solution and in the sensitivity analyses, not used for selection.
    """
    adm = [s for s in sweep if s["min_cluster"] >= min_cluster]
    best = max(adm, key=lambda s: (s["umass"], s["silhouette"]))
    return best["K"], dict(admissible=[s["K"] for s in adm], best_umass=best["umass"], min_cluster=min_cluster,
                           umass_by_K={s["K"]: s["umass"] for s in sweep})


def fit_topics(E, Xc, Xbin, docfreq, idf, vocab, df, K, n_init=50, seed=SEED):
    km = KMeans(K, n_init=n_init, random_state=seed).fit(E)
    lab = km.labels_
    orders = ctfidf_order(Xc, lab, idf, K)
    coh = coherence_umass(Xbin, docfreq, orders)
    centroids = normalize(np.vstack([E[lab == t].mean(0) for t in range(K)]))
    rows = []
    for t in range(K):
        m = lab == t
        near = np.argsort(-(E[m] @ centroids[t]))
        rows.append(dict(
            topic_id=t, n=int(m.sum()), pct=round(100 * m.sum() / len(df), 1),
            coherence=round(float(coh[t]), 3),
            keywords="; ".join(vocab[orders[t][:10]]),
            keywords20="; ".join(vocab[orders[t][:20]]),
            representative_docs=" | ".join(
                str(x)[:110] for x in df[m].iloc[near[:15]]["Article Title"])))
    return lab, orders, coh, centroids, rows


def bh_fdr(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 1.0
    for rank, i in enumerate(order[::-1]):
        running = min(running, ps[i] * m / (m - rank))
        adj[i] = running
    return adj


def trends(df, K, rows, y0=2005, y1=2025):
    yrs = np.arange(y0, y1 + 1)
    sub = df[df["Publication Year"].isin(yrs)]
    tot = sub.groupby("Publication Year").size().reindex(yrs, fill_value=0)
    trend = []
    for t in range(K):
        cnt = sub[sub.topic == t].groupby("Publication Year").size().reindex(yrs, fill_value=0)
        prop = (cnt / tot.replace(0, np.nan)).fillna(0).values * 100
        slope, _, r, p, se = stats.linregress(yrs, prop)
        trend.append(dict(topic_id=t, share_pct=rows[t]["pct"], _slope=slope,
                          b_per_decade=round(slope * 10, 2), t=round(slope / se, 2),
                          p=float(p), r2=round(r ** 2, 3)))
    adj = bh_fdr([x["p"] for x in trend])
    for i, x in enumerate(trend):
        x["p_fdr"] = round(float(adj[i]), 6)
        x["p"] = round(x["p"], 6)
        sl = x.pop("_slope")
        x["classification"] = ("Hot" if sl > 0 and x["p_fdr"] < .05 else
                               "Cold" if sl < 0 and x["p_fdr"] < .05 else
                               "positive (n.s.)" if sl > 0 else "negative (n.s.)")
    return trend, int(len(sub))


def diagnostics(E, lab, centroids, df, K):
    aris = [adjusted_rand_score(lab, KMeans(K, n_init=50, random_state=r).fit_predict(E))
            for r in (1, 7, 13, 21, 99)]  # same settings as the final fit (n_init = 50)
    rng = np.random.default_rng(SEED)
    conc = []
    for rep_i in range(20):
        hold = rng.choice(len(E), size=int(.10 * len(E)), replace=False)
        keep = np.setdiff1d(np.arange(len(E)), hold)
        km2 = KMeans(K, n_init=20, random_state=rep_i).fit(E[keep])
        c2 = normalize(np.vstack([E[keep][km2.labels_ == t].mean(0) for t in range(K)]))
        ri, ci = linear_sum_assignment(-(c2 @ centroids.T))
        mp = dict(zip(ri, ci))
        pred = np.array([mp[t] for t in np.argmax(normalize(E[hold]) @ c2.T, axis=1)])
        conc.append(float((pred == lab[hold]).mean()))
    median_year = df["Publication Year"].median()
    subset = {}
    for name, mask in (("early", df["Publication Year"] <= median_year),
                       ("late", df["Publication Year"] > median_year)):
        Es = E[mask.values]
        kms = KMeans(K, n_init=20, random_state=SEED).fit(Es)
        cs = normalize(np.vstack([Es[kms.labels_ == t].mean(0) for t in range(K)]))
        sim = cs @ centroids.T
        subset[name] = {int(t): round(float(sim[:, t].max()), 3) for t in range(K)}
    recovered = sum(1 for t in range(K)
                    if subset["early"][t] > 0.60 and subset["late"][t] > 0.60)
    # null baselines for the silhouette: (i) random labels of the same sizes; (ii) k-means on data with every
    # embedding dimension permuted independently across documents (marginals kept, dependence destroyed)
    rng = np.random.default_rng(SEED)
    null_sil = [float(silhouette_score(E, rng.permutation(lab), random_state=SEED))
                for _ in range(10)]
    null_shuf = []
    for i in range(10):
        Es = np.column_stack([rng.permutation(E[:, j]) for j in range(E.shape[1])])
        ls = KMeans(K, n_init=10, random_state=i).fit_predict(Es)
        null_shuf.append(float(silhouette_score(Es, ls, random_state=SEED)))
    return dict(silhouette=round(float(silhouette_score(E, lab, random_state=SEED)), 4),
                silhouette_null_mean=round(float(np.mean(null_sil)), 4),
                silhouette_null_sd=round(float(np.std(null_sil)), 4),
                silhouette_shuffled_mean=round(float(np.mean(null_shuf)), 4),
                silhouette_shuffled_sd=round(float(np.std(null_shuf)), 4),
                ari_mean=round(float(np.mean(aris)), 3), ari_min=round(float(np.min(aris)), 3),
                ari_all=[round(a, 3) for a in aris],
                leave_p_out_mean=round(float(np.mean(conc)), 3),
                leave_p_out_sd=round(float(np.std(conc)), 3),
                subset_stability=subset, subset_topics_recovered=int(recovered),
                median_year_split=float(median_year))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--kmax", type=int, default=18)
    ap.add_argument("--k", type=int, default=0, help="0 = apply select_k rule")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)

    df, cascade, dup_log, screened_out = build_corpus(args.input)
    print("Screening cascade:", json.dumps(cascade, indent=1))
    dup_log.to_csv(f"{out}/duplicates.csv", index=False)
    screened_out.to_csv(f"{out}/screened_out.csv", index=False)

    docs = make_docs(df)
    E, Xc, vocab = represent(docs)
    docfreq = np.asarray((Xc > 0).sum(0)).ravel()
    idf = np.log(1 + Xc.shape[0] / (docfreq + 1))
    Xbin = (Xc > 0).astype(np.int8).toarray()
    print(f"n = {Xc.shape[0]} documents, vocabulary = {len(vocab)} terms")

    sweep = k_sweep(E, Xc, Xbin, docfreq, idf, args.kmin, args.kmax)
    sweep_below = k_sweep(E, Xc, Xbin, docfreq, idf, 2, args.kmin - 1) if args.kmin > 2 else []  # coarser solutions, reported for transparency
    for s in sweep:
        print(f"  K={s['K']:2d}  silhouette={s['silhouette']:.4f}  C_UMass={s['umass']:.4f}  "
              f"min_cluster={s['min_cluster']}")
    K, rule = (args.k, {"forced": True}) if args.k else select_k(sweep)
    print("Selected K =", K, rule)

    lab, orders, coh, centroids, rows = fit_topics(E, Xc, Xbin, docfreq, idf, vocab, df, K)
    df["topic"] = lab
    pd.DataFrame(rows).to_csv(f"{out}/table1_topics.csv", index=False)

    trend, n_trend = trends(df, K, rows)
    pd.DataFrame(trend).to_csv(f"{out}/table2_trends.csv", index=False)

    diag = diagnostics(E, lab, centroids, df, K)
    diag.update(dict(screening=cascade, K=K, K_rule=rule, K_sweep=sweep, K_sweep_below=sweep_below, K_range=[args.kmin, args.kmax],
                     umass_mean=round(float(coh.mean()), 4), n_trend_window=n_trend,
                     vocabulary_size=int(len(vocab))))
    json.dump(diag, open(f"{out}/diagnostics.json", "w"), indent=1)

    df[["Article Title", "Publication Year", "topic"]].to_csv(
        f"{out}/doc_assignments.csv", index=False)
    df.to_csv(f"{out}/corpus_final.csv", index=False)
    np.save(f"{out}/E.npy", E)
    np.save(f"{out}/centroids.npy", centroids)

    with open(f"{out}/relevance_lexicon.txt", "w") as fh:
        fh.write("MATCHING: each entry is matched case-insensitively on the lower-cased title + abstract + keywords as a\n"
                 "word-initial stem (\\b + stem; e.g. 'mental' matches 'mental', 'mentally', not 'experimental');\n"
                 "multi-word entries are matched as phrases; context exclusions apply to five entries (see below).\n\n")
        fh.write("PSYCHOLOGICAL / BEHAVIORAL LEXICON\n")
        fh.write("\n".join(lexicon_entries(PSYCH_LEXICON)) + "\n\n")
        fh.write("STRICT PSYCHOLOGICAL LEXICON (sensitivity analysis)\n")
        fh.write("\n".join(lexicon_entries(PSYCH_LEXICON_STRICT)) + "\n\n")
        fh.write("PROCEDURAL-MEDICINE LEXICON\n")
        fh.write("\n".join(lexicon_entries(PROCEDURAL_LEXICON)) + "\n\n")
        fh.write("BASEBALL (TOPICALITY) LEXICON\n")
        fh.write("\n".join(lexicon_entries(BASEBALL_LEXICON)) + "\n\n")
        fh.write("TOPICALITY RULE\n")
        fh.write("retain if (baseball-specific mentions in title >= 1) or (baseball-specific mentions in abstract + keywords >= 2);\n"
                 "a mention is one baseball-specific phrase: the sport name (baseball, softball) or a role/league term (pitcher, batter,\n"
                 "hitter, ballplayer, ballpark, MLB, major/minor/little league, NPB, KBO); a league name combined with the sport name and\n"
                 "its acronym ('Major League Baseball (MLB)', 'Little League baseball') and a sport name followed by a role term\n"
                 "('baseball pitchers') each count once; 'baseball cap(s)' is not a mention\n")
        fh.write("REGEX AS APPLIED (topicality)\n" + BASEBALL_MENTION + "\n\n")
        fh.write("PUBLICATION-TYPE EXCLUSION\n")
        fh.write("title patterns: " + NON_RESEARCH_TITLE + "\n")
        fh.write("plus records whose PubMed publication type is Editorial, Letter, News, Interview, Biography or Portrait (non_research_records.csv)\n\n")
        fh.write("CONTEXT EXCLUSIONS\n")
        for k, v in LEXICON_CONTEXT.items():
            fh.write(f"{k}: {v}\n")
        fh.write("\nDECISION RULE\n")
        fh.write("retain if (psychological matches >= 2) and "
                 "(psychological matches >= procedural matches)\n\n")
        fh.write("REGEX AS APPLIED (psychological lexicon)\n" + compile_lexicon(PSYCH_LEXICON) + "\n\n")
        fh.write("REGEX AS APPLIED (procedural-medicine lexicon)\n" + compile_lexicon(PROCEDURAL_LEXICON) + "\n")

    print(f"\nsilhouette            : {diag['silhouette']:.4f} (null {diag['silhouette_null_mean']:.4f})")
    print(f"C_UMass (mean)        : {coh.mean():.4f}")
    print(f"ARI over 5 seeds      : mean {diag['ari_mean']:.3f}, min {diag['ari_min']:.3f}")
    print(f"leave-p-out (20 reps) : mean {diag['leave_p_out_mean']:.3f}, SD {diag['leave_p_out_sd']:.3f}")
    print(f"subset stability      : {diag['subset_topics_recovered']}/{K} topics recovered in both halves")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
