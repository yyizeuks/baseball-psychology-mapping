# Mapping the Knowledge Structure of Baseball Psychology Research

Code, data, and results for:

> Lee, J. W., and Han, D. H. Mapping the knowledge structure of baseball psychology research: a systematic mapping review using topic modeling and an auditable relevance screen. *Frontiers in Psychology* (under review).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24050512.svg)](https://doi.org/10.5281/zenodo.24050512)

Repository: https://github.com/yyizeuks/baseball-psychology-mapping. The release archived on Zenodo and cited in the article's Data Availability Statement is v1.2.0, doi: [10.5281/zenodo.24050512](https://doi.org/10.5281/zenodo.24050512) (see also `CITATION.cff`).

Every statistic, table, and figure in the manuscript is regenerated from the retrieved records by the scripts in `code/` under a fixed random seed (42). Run the scripts with `PYTHONHASHSEED=0` so that the ordering of tied keyword counts is identical across runs.

## Reproduce

```bash
pip install -r code/requirements.txt
cd code
export PYTHONHASHSEED=0
python analysis_pipeline_v2.py --input ../data/baseball_psychology_retrieved_records.xlsx   # corpus (two-sided screen), LSA representation, K sweep and rule, topics, trends, diagnostics -> out/
python analysis_sensitivity_v2.py        # K range, rule seeds, SVD dimensions, bootstrap, strict lexicons, trend estimators, domains, seeds, sub-structure, emerging keywords
python apply_labels.py                   # writes the domain membership recorded in labels.json (from the 'domains' block) into the corpus files
python enrich_pubmed.py                  # PubMed metadata (network access required; ~35 min) -> out/enriched_pubmed.csv
python enrich_mesh.py                    # MeSH headings for all eligible records (network access required)
python validate_mesh.py                  # comparison of the screen with indexer-assigned headings
python bibliometrics_v2.py               # bibliometric profile, co-word, country, author and MeSH networks; manuscript Figures 4-5, Supplementary Figures S2-S3
python analysis_biomech_sens.py          # psychology-core screen (biomechanical studies excluded) and post hoc removal of domain C
python derived_stats.py                  # domain and within-domain trends, strand trends, label checks, venue concentration, source-composition check, representative documents
python figures_v2.py                     # manuscript Figures 1-3 and 6-9, Supplementary Figures S1 and S4-S7
python export_tiff.py                    # TIFF export of the manuscript figures in manuscript numbering
```

`remap_enrichment.py` re-keys an existing enrichment cache to a rebuilt corpus so that PubMed does not have to be queried again after a change to the screen.

Expected runtime on a 2-core machine: about 30 minutes excluding the PubMed and MeSH enrichment.

## Figure numbering

Figures are numbered in the manuscript in order of first mention. The source files keep their generation names:

| Manuscript | Source file | | Supplementary | Source file |
|---|---|---|---|---|
| Figure 1 | fig1_pipeline.png | | Figure S1 | figS1_abstract_period.png |
| Figure 2 | fig2_prisma.png | | Figure S2 | figS2_coauthors.png |
| Figure 3 | fig3_corpus.png | | Figure S3 | figS3_mesh.png |
| Figure 4 | fig4_countries.png | | Figure S4 | figS4_stability_lexicon.png |
| Figure 5 | fig5_coword.png | | Figure S5 | figS5_domains.png |
| Figure 6 | fig6_selection.png | | Figure S6 | figS6_substructure.png |
| Figure 7 | fig7_landscape.png | | Figure S7 | figS7_krange.png |
| Figure 8 | fig8_trends.png | | | |
| Figure 9 | fig9_domains.png | | | |

## Contents

| Path | Description |
|---|---|
| `code/analysis_pipeline_v2.py` | Retrieval -> de-duplication (accent-insensitive normalized titles) -> publication type -> completeness -> language -> window -> topicality rule (phrase-level baseball mentions) -> relevance screen (word-initial stem matching with context exclusions); LSA representation; K sweep with the explicit selection rule (admissible solution with the highest C_UMass); topics; trends (OLS + BH-FDR); diagnostics |
| `code/analysis_sensitivity_v2.py` | Sensitivity blocks: K = 5-14 (K = 5 lies below the pre-specified range), rule across sweep seeds, SVD dimension (with topic and domain trends), Hennig bootstrap (100 resamples; topics and domains), strict lexicons, binomial/CLR/Mann-Kendall/quadratic/era trend tests, hierarchical domains with bootstrap co-grouping, five alternative initializations at the selected K, sub-structure, emerging keywords |
| `code/apply_labels.py` | Writes the data-derived domain membership recorded in `labels.json` into the corpus files (single source for all downstream scripts) |
| `code/analysis_biomech_sens.py` | Psychology-core screen (biomechanical-measurement exclusion lexicon) and post hoc removal of the biomechanical domain |
| `code/derived_stats.py` | Domain-level and within-domain trends under four estimators, strand-level trends, label checks, venue concentration, source-composition check (PubMed-matched vs unmatched), venue-novelty proxy, representative documents |
| `code/enrich_pubmed.py`, `code/enrich_mesh.py`, `code/remap_enrichment.py`, `code/validate_mesh.py` | Title matching to PubMed (E-utilities): journal, authors, affiliations, MeSH, cited-in counts; MeSH tree numbers; comparison of the screen with indexer-assigned psychology headings |
| `code/bibliometrics_v2.py` | Productivity, citation impact, h-index, author/country/institution collaboration indicators and networks, co-word analysis (Louvain), venue overlap |
| `code/figures_v2.py`, `code/export_tiff.py` | Manuscript and supplementary figures; TIFF export |
| `code/labels.json` | Topic labels and data-derived domain membership (author-assigned labels; domains from hierarchical clustering) |
| `data/baseball_psychology_retrieved_records.xlsx` | The 1,848 records retrieved from Crossref, PubMed, and the Web of Science (title, year, abstract, keywords) |
| `data/corpus_screened_with_topics.csv` | The 671 screened articles with topic and domain assignments and lexicon match counts |
| `data/records_excluded_by_topicality_or_relevance.csv` | The 268 records excluded by the topicality rule and the 454 excluded by the relevance screen, with baseball-mention and lexicon match counts |
| `data/duplicate_groups_collapsed.csv` | The duplicate groups collapsed by the normalized de-duplication (196 records removed) |
| `data/non_research_records.csv` | Records matched to PubMed whose publication type is Editorial, Letter or Biography (excluded by the publication-type step together with title-pattern matches) |
| `data/topicality_audit.csv` | Reading of the 84 records admitted by the topicality rule on exactly two abstract/keyword mentions and none in the title (baseball-specific / multi-sport / incidental) |
| `data/relevance_lexicon.txt` | The baseball lexicon and topicality rule; the psychological, strict-psychological, and procedural-medicine lexicons with context exclusions; the regular expressions as applied; the decision rule |
| `data/pubmed_metadata_enrichment.csv` | PubMed metadata for the 415 matched articles (and match status for all 671) |
| `data/mesh_validation_data.json` | MeSH headings (descriptor UIs, qualifiers, major-topic flags) and tree numbers for all PubMed-matched eligible records |
| `data/representative_documents.csv` | The 15 articles closest to each topic centroid, the evidence behind each topic label |
| `results/` | All intermediate results (diagnostics, sweeps, sensitivity blocks, bibliometric profile, derived statistics, tables, document vectors and centroids) |
| `figures/` | PNG versions of all figures |

## Data terms

Bibliographic records were retrieved from Crossref, PubMed, and the Web of Science Core Collection. The deposited files contain titles, publication years, abstracts, keywords, and derived fields only; redistribution of Web of Science records is subject to Clarivate's terms of use, and users who re-use the retrieved records should consult those terms. PubMed metadata are in the public domain.

## License

Code: MIT License. Derived data and figures: CC BY 4.0.
