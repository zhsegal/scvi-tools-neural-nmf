# State-of-the-Art Pretrained Gene Embeddings — A Survey for Use as a Semantic Prior

*Compiled: May 2026. Audience: SemanticSCVI / scviva fork. Goal: replace or augment the Geneformer-derived gene distance matrix used as a semantic prior with the strongest currently-available alternative. Optimization: (1) embedding quality, (2) ease of access.*

---

## TL;DR — What to Use

If you only read one paragraph: **switch your default semantic prior to GenePT's `text-embedding-3-large` embeddings of NCBI gene summaries** (3072-d, ~33k human HGNC genes, one Zenodo pickle download, MIT/CC-BY, no GPU needed). It is the single embedding family with the strongest, most independently-replicated evidence of producing biologically meaningful gene-gene distances, and it has already been validated as a VAE prior by sciLaMA (Feb 2025). For complementary signal, stack it with **scPRINT-2's ESM-2-derived gene token table** (Dec 2025 release, on CZI Virtual Cells Platform) which encodes protein-sequence similarity. If you want to stay within the same family as your current setup, **Geneformer-V2-104M** (Dec 2024, Apache-2.0 on HuggingFace) is a strict, drop-in upgrade over Geneformer-V1.

The most counterintuitive finding from 2024–2026 benchmarks is that the giant scRNA-trained foundation models (scGPT, Geneformer, scFoundation, UCE) **do not clearly produce better gene-gene distances than literature-derived LLM embeddings or even classical co-expression baselines** on the kinds of tasks your prior is meant to support. Several papers — Kedzierska 2025 (Genome Biology), Ahlmann-Eltze 2025 (Nature Methods), Boiarsky 2023, and the Zhong 2025 benchmark of 38 embedding methods — all point in the same direction: simpler is often better at the gene level. This shapes the recommendation below.

---

## Why You Should Reconsider the SCFM (Single-Cell Foundation Model) Default

Geneformer, scGPT, scFoundation, UCE and CellPLM dominate Twitter/X discussion and have polished tutorials, so it is easy to assume one of them is the right default. Recent independent benchmarks complicate that assumption, and they matter specifically for your use case (a *distance matrix between genes*, not cell-level embeddings):

The Zhong et al. 2025 benchmark from the Yao Lab at Rice (bioRxiv 2025.01.29.635607, evaluating 38 gene embedding methods across single-gene attributes, paired gene interactions, and pathway recovery) concluded that biomedical-literature embeddings consistently dominate the SCFMs on general predictive tasks; protein-sequence embeddings dominate on functional/genetic-interaction tasks; expression embeddings (where SCFMs would compete) are best only for disease-related tasks. Crucially, performance on these gene-level tasks "does not scale with model complexity."

Kedzierska et al. 2025 (Genome Biology) found that in zero-shot settings Geneformer and scGPT often underperform scVI, Harmony, and even HVG-only baselines on cell-clustering tasks; Boiarsky et al. 2023 (MIT, bioRxiv) showed L1 logistic regression matches or beats scBERT for cell-type annotation even in the few-shot regime; and Ahlmann-Eltze et al. 2025 (Nature Methods) tested five foundation models on gene perturbation effect prediction and found **none beat simple linear baselines**.

The most directly analogous prior work to yours, sciLaMA (Wang et al., bioRxiv 2025.01.28.635153, Feb 2025), explicitly injects precomputed static gene embeddings from multimodal LLMs into a siVAE-style architecture and reports that this regularizes the model and beats scVI. So the design pattern of using a static external gene embedding matrix as a prior in a VAE is already on solid ground — what they used is essentially the GenePT route, not the SCFM route.

This does not mean SCFMs are useless; it means that for a *gene similarity prior* specifically, the burden of proof falls on the heavy transformer, not on the simpler text/sequence embedding.

---

## The Full Landscape — Five Families of Gene Embeddings

To make the comparison digestible, it helps to group what's available into five conceptual families. Each family is sourced from a fundamentally different signal, which matters because the distances they produce mean different things.

**Family A — Literature-derived embeddings.** A general-purpose LLM (OpenAI ada-002, text-embedding-3-large, gpt-4o, or a domain-tuned BERT) encodes the NCBI gene summary, UniProt description, or GO annotations for each gene. Distance ≈ "how similar are the things written about these two genes in the biomedical literature." Examples: GenePT, scELMo, scGenePT, PubMedBERT-on-NCBI. Strengths: full transcriptome coverage (~33k genes), trivially downloadable as a single pickle/parquet, no GPU needed, encodes pathway/function/disease semantics; matches or beats SCFMs on most gene-level benchmarks. Weaknesses: literature-popularity bias (well-studied genes have sharper representations, novel/uncharacterized genes get generic embeddings), cannot represent biology not yet in print.

**Family B — scRNA-trained foundation models (SCFMs).** Transformers trained on tens to hundreds of millions of single-cell transcriptomes, with a gene token vocabulary. Examples: Geneformer-V2, scGPT, scFoundation, CellPLM, CellFM, scLong, scCello. Strengths: distances are grounded in observed co-expression and cell-state context; can be made context-aware by running the model over a reference dataset. Weaknesses: gene vocabulary is usually restricted (19k–60k tokens, often with HGNC/Ensembl quirks), checkpoint sizes are large (often 100M–1B params), and recent benchmarks question whether the gene-level signal is actually richer than literature or sequence baselines.

**Family C — Protein-sequence embeddings used as gene proxies.** Take the canonical protein product of each gene and embed it with ESM-2, ESM-3, or a similar protein language model; mean-pool over residues. Distance ≈ protein-sequence and protein-structure similarity. Examples: ESM-2-650M, ESM-3, ProteinCLIP (DS569k on HF). UCE and scPRINT both literally bootstrap their gene tokens from ESM-2 in this way, so you can capture the same signal without running their full pipelines. Strengths: well-defined, deterministic, captures evolutionary/functional similarity that text-based methods miss for understudied genes. Weaknesses: covers only protein-coding genes (~19k human), misses regulatory non-coding RNAs, and aggregates loss when collapsing residue-level to a single per-gene vector.

**Family D — Co-expression / network embeddings.** Classical baselines: gene2vec (Du et al. 2019, ~17k genes × 200-d trained on bulk co-expression), node2vec / GRN-derived embeddings, PPI-network embeddings (used in PINNACLE). Distance ≈ co-expression or interaction frequency. Strengths: extremely cheap, well-understood, often competitive on paired-gene tasks. Weaknesses: low dimensionality, no functional semantics beyond co-occurrence patterns.

**Family E — Knowledge-graph and ontology-informed embeddings.** Models that explicitly bake biological knowledge into their pretraining objective. Examples: GeneCompass (GRN + promoter + gene-family priors built into a 120M-cell transformer), OntoProtein (GO-aware protein embeddings), PINNACLE (context-specific protein embeddings, 156 cell types × PPI signal). Strengths: distances explicitly encode the kind of structured biological knowledge a semantic prior is meant to enforce; arguably the most philosophically aligned with your use case. Weaknesses: smaller community than Families A and B; some are awkward to extract a clean per-gene matrix from.

---

## Detailed Per-Model Profiles

### Tier 1 — Recommended (download today, plug in tonight)

**GenePT — Stanford, Chen & Zou (Family A).** The reference implementation of "encode NCBI gene summaries with a general LLM." Two pre-computed flavors live on Zenodo (record 10833191): the original ada-002 version (1536-d, ~33k HGNC symbols) and the March 2024 update using `text-embedding-3-large` over NCBI + UniProt summaries (3072-d). Published Nature Biomedical Engineering 2024. Code at github.com/yiqunchen/GenePT, MIT license, data CC-BY. To re-embed yourself with a newer model costs roughly $5–10 in OpenAI API calls for the full human gene set. Quality: in the paper, ada-002 embeddings tie or beat Geneformer on 4 of 9 gene property tasks and are competitive with scGPT throughout; in Zhong et al. 2025 the literature-embedding family as a whole beats both Geneformer and scGPT on most general gene tasks. This is the strongest "best evidence per dollar of compute" choice, and the easiest to defend in a paper.

**scPRINT-2 — Cantini Lab, Kalfon et al. (Family B/C hybrid).** v1 published Nature Communications 2025; v2 preprint Dec 2025 expanded to 350M cells across 22k datasets and 16 species. The reason it stands out: scPRINT exposes its per-gene embeddings as an explicit parquet artifact (see `notebooks/generate_gene_embeddings.ipynb`), and the gene tokens are themselves seeded from ESM-2 protein embeddings — so you get both Family B's expression context and Family C's protein-sequence signal in one matrix. MIT-licensed, weights distributed via the CZI Virtual Cells Platform (virtualcellmodels.cziscience.com/model/scprint). Pip-installable. Designed for GRN inference, so the embeddings encode regulatory/functional relationships rather than only co-expression. Of all the new SCFMs, this has the cleanest path from `pip install` to a fixed `(n_genes, d)` matrix on disk.

**Geneformer-V2-104M — Theodoris lab, Broad/Gladstone (Family B).** December 2024 release on HuggingFace (`ctheodoris/Geneformer`, also mirrored at `nvidia/geneformer_V2_104M` and `nvidia/geneformer_V2_316M`). Trained on Genecorpus-104M, human-only, 20k protein-coding gene vocab, Apache-2.0. Same `EmbExtractor` API as V1, so if your pipeline already consumes Geneformer embeddings, V2 is a literal one-line swap. Hidden dim 512 (V2-104M) or 896 (V2-316M). Static token-embedding matrix is accessible directly at `model.bert.embeddings.word_embeddings.weight` if you want the context-free version, or via `EmbExtractor(emb_mode="gene", emb_layer=-1)` for context-pooled embeddings over a reference dataset. The Helical.bio benchmark (and several others) shows V2 substantially outperforms V1; treat this as the bare minimum upgrade if you do nothing else.

**ESM-2 protein embeddings — Meta (Family C).** Not a gene model per se, but for protein-coding genes it gives one of the strongest, most-tested embeddings available. The 650M variant (`facebook/esm2_t33_650M_UR50D`) gives 1280-d per-residue vectors; mean-pool over the canonical UniProt isoform to get one vector per gene. The Zhong 2025 benchmark found sequence embeddings (i.e. ESM-class) are the single best signal for functional and genetic-interaction prediction tasks. Coverage is ~19k human protein-coding genes; you lose non-coding RNAs entirely. The major attraction is that this is *orthogonal* to GenePT — protein sequence similarity and biomedical literature similarity capture different things — so concatenating ESM-2 with GenePT is the highest-EV "stacked semantic prior" you can build without much engineering.

### Tier 2 — Strong but with friction

**scGPT — Bo Wang lab (Family B).** ~51M params, 33M human cells, Nature Methods 2024. MIT license, mature `pip install scgpt`, and the gene token embedding matrix is directly accessible at `model.encoder.weight` (shape `(~60k, 512)`). The drawback is checkpoint distribution: weights are on Google Drive (not HuggingFace), which periodically breaks. If you can tolerate that, this is the most mature SCFM ecosystem and the official GRN tutorial demonstrates exactly the "extract gene embedding matrix from a transformer" pattern you want.

**scFoundation / xTrimoGene — BioMap + Tsinghua (Family B).** 100M params, 50M human cells, 19,264 protein-coding gene vocab, Apache-2.0 code (separate "Model License" for the weights — read it before commercial use). On HuggingFace at `genbio-ai/scFoundation`. Unique selling point: continuous magnitude-aware expression encoding (no binning), which matters more for cell embeddings than for a gene prior. The catch for your use case: gene identity is encoded positionally rather than as a token-embedding lookup, so extracting a static per-gene matrix requires running the decoder over a reference dataset and averaging. Worth the effort only if you have a specific reason to want a magnitude-aware embedding.

**Nicheformer — Theis lab (Family B with spatial signal).** Nature Methods 2025, 110M cells (57M dissociated + 53M spatial), human + mouse. MIT, clean HuggingFace release at `theislab/Nicheformer`, code at `github.com/theislab/nicheformer`. Token embedding table is directly accessible. The unique value for *your* fork specifically: it has been pretrained with spatial-niche context, which synergizes with your scviva extension. If you want a single embedding source that aligns with both SemanticSCVI and SCVIVA, this is the natural choice.

**GeneCompass — Cell Research 2024, Yang et al. (Family E).** 120M cells (human + mouse), explicitly knowledge-informed: GRN structure, promoter information, and gene family relationships are built into the pretraining objective. Apache-2.0, repo at `github.com/xCompass-AI/GeneCompass`. Of all the SCFMs, this is the most "semantic by design" — the distances between gene embeddings already encode the structured biological knowledge a semantic prior is meant to enforce. Less ecosystem polish than scGPT or Geneformer, but on paper the most aligned with your goal.

**scELMo — Liu et al., Yale (Family A).** A direct refinement of GenePT using gpt-4o text embeddings over richer per-feature descriptions. Published Cell Patterns 2025. The authors report it outperforms GenePT, scGPT, and Geneformer on clustering and cell-type annotation. Pre-computed gene embeddings on `github.com/HelloWorldLTY/scELMo` (Google Drive link in README), MIT license. If you want the GenePT idea with a slightly newer LLM and a published "we beat the previous version" claim, this is the upgrade.

**scGenePT — CZI Virtual Cells Platform (Family A).** Releases three separate pre-computed text-embedding-3-large vectors per gene: one from NCBI descriptions, one from UniProt summaries, and one split into the three GO axes (Molecular Function, Biological Process, Cellular Component). The unique value is that you can stack multiple complementary semantic *views* of each gene and let your model decide which matters per latent factor. Hosted at virtualcellmodels.cziscience.com/model/scgenept.

### Tier 3 — Interesting but not the top pick

**CellFM** (Nature Communications 2025, 800M params, 100M human cells) reports the strongest gene-relationship benchmark numbers among the new SCFMs but is built on MindSpore rather than PyTorch, which is meaningful friction for a PyTorch-Lightning codebase like yours. **scLong** (Nature Communications 2026, 1B params, full 28k-gene context) is the largest currently-released model and worth watching, but repo maturity should be confirmed before committing. **scCello** (NeurIPS 2024 Spotlight) is ontology-aware and smaller (good for fast iteration). **CellPLM** (ICLR 2024, BSD-2-Clause, ~80M params) is fast and offers spatial pretraining, but its generalization to held-out tissues has been mixed in independent benchmarks. **UCE** (650M params, Stanford-Snap) gives cell embeddings primarily; its gene tokens are simply ESM-2 protein embeddings, so for a gene prior you should use ESM-2 directly rather than running the full 650M pipeline. **PubMedBERT on NCBI summaries** (`NeuML/pubmedbert-base-embeddings`, 768-d, free, runs on CPU) is the obvious local fallback if you ever want to avoid OpenAI API costs while staying in Family A — Chen et al. 2025 (Comp. Struct. Biotech. J.) showed it matches OpenAI ada-002 on several benchmark tasks.

### Tier 4 — Skip

**scBERT** is historically important but obsolete; multiple independent benchmarks show it loses to both scGPT and simpler baselines. **scMulan** (Tsinghua, 368M params) is less actively maintained and not gene-embedding-focused. **Cell2Sentence / C2S-Scale 27B** treats genes as text tokens inside a Gemma-2 27B language model, which conflates gene identity with general language tokens and provides no clean gene-embedding matrix. **Evo-2** (Arc Institute, 40B params, 9T nucleotides) is genome-sequence-level rather than gene-indexed, so extracting one vector per gene requires arbitrary pooling decisions. **scMASTER / scMOST** did not surface as released models — likely paper-only or aspirational.

---

## Comparison Table

| Model | Family | Params | Gene vocab | License | Static gene matrix? | Access | Best evidence quality |
|---|---|---|---|---|---|---|---|
| **GenePT (text-3-large)** | A: Literature | n/a (uses OpenAI API) | ~33k HGNC | MIT + CC-BY | Yes — pickle | Zenodo 10833191 | **Strongest, multiple independent benchmarks** |
| **scPRINT-2** | B/C hybrid | 50M+ | ESM-2 derived | MIT | Yes — parquet | CZI Virtual Cells Platform | Strong, GRN-focused |
| **Geneformer-V2-104M** | B: SCFM | 104M | ~20k Ensembl | Apache-2.0 | Yes — `word_embeddings.weight` | HuggingFace `ctheodoris/Geneformer` | Good (V2 ≫ V1) |
| **ESM-2-650M** | C: Protein | 650M | ~19k UniProt | MIT | Yes (after mean-pool) | HuggingFace `facebook/esm2_t33_650M_UR50D` | Best for functional / genetic interaction tasks |
| **scELMo** | A: Literature | n/a | ~33k HGNC | MIT | Yes — Google Drive | github.com/HelloWorldLTY/scELMo | Beats GenePT in paper |
| **scGenePT** | A: Literature (3 views) | n/a | ~33k HGNC | (CZI terms) | Yes — multiple files | CZI Virtual Cells Platform | Strong, multi-view |
| **scGPT (whole-human)** | B: SCFM | 51M | ~60k | MIT | Yes — `encoder.weight` | Google Drive (fragile) | Competitive on cell tasks |
| **scFoundation** | B: SCFM | 100M | 19,264 | Apache + model lic. | No — needs inference | HF `genbio-ai/scFoundation` | Mixed on gene tasks |
| **Nicheformer** | B: SCFM (spatial) | — | — | MIT | Yes — token table | HuggingFace `theislab/Nicheformer` | Strong for spatial |
| **GeneCompass** | E: Knowledge-aware | — | human+mouse | Apache-2.0 | Yes | github.com/xCompass-AI/GeneCompass | Strong by design |
| **CellFM** | B: SCFM | 800M | — | (research) | Yes | github.com/biomap-research/CellFM | Best new SCFM numbers, MindSpore friction |
| **PubMedBERT on NCBI** | A: Literature (free) | 110M | ~33k HGNC | MIT | Yes (run yourself) | HuggingFace `NeuML/pubmedbert-base-embeddings` | Matches ada-002, free, CPU-OK |
| **gene2vec** | D: Co-expression | n/a | ~17k | research | Yes — `.npy` | Du et al. 2019 GitHub | Solid classical baseline |
| **scBERT** | B: SCFM (legacy) | 5M | — | — | Yes (gene2vec init) | TencentAILabHealthcare/scBERT | **Skip — obsolete** |
| **UCE-650M** | B: SCFM (cell) | 650M | cross-species | MIT | No (just use ESM-2) | github.com/snap-stanford/UCE | Cell-level, not gene-level |

---

## Concrete Recommendation for SemanticSCVI

For the SemanticSCVI / SCVIVA fork as it stands today, my recommendation is a three-step plan rather than a single replacement.

**Step 1 — Replace your current Geneformer-V1 distance matrix with GenePT-3-large.** Download the pickle from Zenodo 10833191, take pairwise cosine distance on the 3072-d vectors, restrict to the gene set in your reference AnnData. This is one file download and roughly fifty lines of NumPy. The empirical evidence (Zhong 2025, GenePT paper, sciLaMA precedent) suggests this will produce more biologically meaningful gene-gene distances than your current Geneformer prior, with effectively zero compute cost and the convenient property that every gene in your dataset is covered (no missing-token problem). This alone is likely the single highest-impact change.

**Step 2 — Add ESM-2 protein embeddings as a second, orthogonal semantic view.** Either compute pairwise distances from ESM-2 and combine the two distance matrices (e.g. weighted average, or concatenate then re-normalize), or feed both as separate semantic maps your SemanticMogPrior can attend over. Family A and Family C capture genuinely different similarity structure (literature semantics vs. protein-sequence semantics), so the combined prior will be strictly more informative than either alone. Coverage drops to protein-coding genes for the ESM component, which you handle by zero-imputing or falling back to GenePT for non-coding genes.

**Step 3 — Compare against a SCFM-derived prior using your benchmark suite.** Pick *one* SCFM as a head-to-head comparison: **Geneformer-V2-104M** if you want the simplest in-family upgrade, **scPRINT-2** if you want the best of the new transformers, or **Nicheformer** if you want spatial alignment with SCVIVA. Run your `SemanticBenchmark.run_all()` and `SemanticSpatialBenchmark.benchmark_niche_coherence()` against each prior choice and report ELBO + niche coherence + pathway recovery side-by-side. Given the benchmark literature, I expect the SCFM-derived prior to roughly match GenePT+ESM-2 rather than dominate it, but you will not know until you test it on *your* data and *your* downstream metrics.

The reason for that ordering is straightforward: GenePT is essentially free to swap in and has the best independent evidence; ESM-2 stacking is cheap and captures complementary signal; running an SCFM comparison is the highest-effort step and should be undertaken only once the cheaper options are baselined.

---

## Practical Caveats

A few things worth knowing before you commit to any of these.

Literature embeddings have a well-documented *popularity bias*: TP53, BRCA1, and other heavily-studied genes have very sharp, well-differentiated representations, while novel or lncRNA genes get generic embeddings dominated by their HGNC symbol. If your scientific goal is *discovering* novel gene function (rather than enforcing known biology as a prior), this is a real limitation. For SemanticSCVI used as a regularizer the bias is mostly benign — you want known biology to anchor the latent space.

The SCFM benchmarks above (Kedzierska, Boiarsky, Ahlmann-Eltze) primarily evaluate *cell-level* and *perturbation* tasks rather than gene-similarity directly. The Zhong 2025 benchmark is the most direct gene-level evidence and points firmly toward literature and sequence embeddings; treat Zhong as the most relevant data point for your decision.

Several SCFMs (scFoundation, UCE) are *not* easy to extract a static per-gene matrix from — their architectures encode gene identity positionally or via per-cell context. If you go the SCFM route, you will either accept that and run inference over a reference dataset to mean-pool, or you stick to models with explicit token-embedding tables (scGPT, Geneformer, Nicheformer, scPRINT, GeneCompass).

License terms vary. Apache-2.0 (Geneformer-V2, GeneCompass, scFoundation code), MIT (scGPT, scPRINT, Nicheformer, GenePT, scELMo, ESM-2, PubMedBERT) and BSD-2 (CellPLM) are all permissive. scFoundation has a separate "Model License" on the weights — check it before any commercial use. CZI Virtual Cells Platform has its own terms.

A sparse-autoencoder probe of Geneformer and scGPT (arXiv 2603.02952) found that SCFMs encode "organized biological knowledge — pathway membership, protein interactions, functional modules" reasonably well but contain *minimal causal regulatory logic*. The implication for your prior: SCFM embeddings will work as a semantic *similarity* prior but should not be over-interpreted as encoding gene regulatory direction.

---

## Sources

Primary models and weights:
- [GenePT code](https://github.com/yiqunchen/GenePT) — Chen & Zou
- [GenePT pre-computed embeddings on Zenodo](https://zenodo.org/records/10833191)
- [Geneformer on HuggingFace](https://huggingface.co/ctheodoris/Geneformer)
- [Geneformer V2-104M (NVIDIA mirror)](https://huggingface.co/nvidia/geneformer_V2_104M)
- [scGPT repo](https://github.com/bowang-lab/scGPT)
- [scFoundation on HuggingFace](https://huggingface.co/genbio-ai/scFoundation)
- [scPRINT on CZI Virtual Cells Platform](https://virtualcellmodels.cziscience.com/model/scprint)
- [scPRINT repo](https://github.com/cantinilab/scPRINT)
- [Nicheformer on HuggingFace](https://huggingface.co/theislab/Nicheformer)
- [GeneCompass repo](https://github.com/xCompass-AI/GeneCompass)
- [CellFM repo](https://github.com/biomap-research/CellFM)
- [UCE repo](https://github.com/snap-stanford/UCE)
- [CellPLM repo](https://github.com/OmicsML/CellPLM)
- [scELMo repo](https://github.com/HelloWorldLTY/scELMo)
- [scGenePT on CZI Virtual Cells Platform](https://virtualcellmodels.cziscience.com/model/scgenept)
- [ESM-2 on HuggingFace](https://huggingface.co/facebook/esm2_t33_650M_UR50D)
- [PubMedBERT embeddings on HuggingFace](https://huggingface.co/NeuML/pubmedbert-base-embeddings)
- [PINNACLE repo](https://github.com/mims-harvard/PINNACLE)

Benchmark and critical evaluation papers:
- [Zhong et al. 2025 — Benchmarking 38 gene embeddings](https://www.biorxiv.org/content/10.1101/2025.01.29.635607v1)
- [Kedzierska et al. 2025 — Zero-shot limitations of SCFMs (Genome Biology)](https://link.springer.com/article/10.1186/s13059-025-03574-x)
- [Boiarsky et al. 2023 — Deep Dive into scRNA-seq Foundation Models](https://www.biorxiv.org/content/10.1101/2023.10.19.563100v1)
- [Ahlmann-Eltze et al. 2025 — Foundation models do not beat linear baselines on perturbation prediction (Nature Methods)](https://www.nature.com/articles/s41592-025-02772-6)
- [sciLaMA — LLM gene embeddings as VAE prior, Feb 2025](https://www.biorxiv.org/content/10.1101/2025.01.28.635153v2.full.pdf)
- [GenePT preprint](https://www.biorxiv.org/content/10.1101/2023.10.16.562533v2.full)
- [scGPT, Nature Methods 2024](https://www.nature.com/articles/s41592-024-02201-0)
- [scPRINT, Nature Communications 2025](https://www.nature.com/articles/s41467-025-58699-1)
- [CellFM, Nature Communications 2025](https://www.nature.com/articles/s41467-025-59926-5)
- [Helical.bio Geneformer V1 vs V2 benchmark](https://www.helical.bio/blog/benchmarking-geneformer-v1-vs-v2-bio-foundation-models)
- [Sparse-autoencoder probe of Geneformer & scGPT](https://arxiv.org/html/2603.02952v1)
