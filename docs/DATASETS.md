# Datasets and citations

Every task pulls a public dataset. **Cite the dataset authors, not this repo**,
for any result you report — ProtBench only runs the evaluation.

Two tiers below. The *verified* entries were checked against the upstream
release, row counts and all, and carry caveats you need before comparing to
published numbers. The *best-effort* entries name the benchmark the data is
generally known to come from, but have **not** been individually re-verified —
confirm against the HuggingFace card before citing one in a paper.

Many of these are third-party rehosts. A rehost can differ from the original in
splits, filtering or label definition without saying so, which is why the
distinction matters.

## Task index

| Task(s) | HuggingFace dataset | Type | Main metric |
|---|---|---|---|
| `ec_classification` | [`AI4Protein/EC`](https://huggingface.co/datasets/AI4Protein/EC) | multilabel | F1_Micro |
| `go_mf` | [`AI4Protein/GO_MF`](https://huggingface.co/datasets/AI4Protein/GO_MF) | multilabel | F1_Macro |
| `go_bp` | [`AI4Protein/GO_BP`](https://huggingface.co/datasets/AI4Protein/GO_BP) | multilabel | F1_Macro |
| `go_cc` | [`AI4Protein/GO_CC`](https://huggingface.co/datasets/AI4Protein/GO_CC) | multilabel | F1_Macro |
| `profet_np_sp_cleaved` | [`GrimSqueaker/ProFET_NP_SP_Cleaved`](https://huggingface.co/datasets/GrimSqueaker/ProFET_NP_SP_Cleaved) | binary | AUC |
| `signalp_binary` | [`GrimSqueaker/SignalP_Binary`](https://huggingface.co/datasets/GrimSqueaker/SignalP_Binary) | binary | AUC |
| `cath_eat` | [`GrimSqueaker/cath43-eat`](https://huggingface.co/datasets/GrimSqueaker/cath43-eat) | multiclass | Accuracy |
| `proteingym_clinical_indels_supervised`, `proteingym_clinical_indels_zeroshot`, `proteingym_clinical_substitutions_supervised`, `proteingym_clinical_substitutions_zeroshot`, `proteingym_dms_indels_supervised`, `proteingym_dms_indels_zeroshot`, `proteingym_dms_substitutions_supervised`, `proteingym_dms_substitutions_zeroshot` | [`OATML-Markslab/ProteinGym_v1`](https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v1) | binary | AUC |
| `rhla_enzyme_mutations` | [`SaProtHub/DATASET-CAPE-RhlA-seqlabel`](https://huggingface.co/datasets/SaProtHub/DATASET-CAPE-RhlA-seqlabel) | regression | Spearman |
| `aav_flip` | [`SaProtHub/Dataset-AAV-FLIP`](https://huggingface.co/datasets/SaProtHub/Dataset-AAV-FLIP) | regression | Spearman |
| `beta_lactamase_peer` | [`SaProtHub/Dataset-Beta_Lactamase-PEER`](https://huggingface.co/datasets/SaProtHub/Dataset-Beta_Lactamase-PEER) | regression | Spearman |
| `signal_peptide` | [`SaProtHub/Dataset-Signal-Peptides`](https://huggingface.co/datasets/SaProtHub/Dataset-Signal-Peptides) | token_classification | F1_Macro |
| `thermostability` | [`SaProtHub/Dataset-Thermostability-FLIP`](https://huggingface.co/datasets/SaProtHub/Dataset-Thermostability-FLIP) | regression | Spearman |
| `ppi_bernett` | [`Synthyra/bernett_gold_ppi`](https://huggingface.co/datasets/Synthyra/bernett_gold_ppi) | binary | AUC |
| `disorder`, `ss3` | [`agemagician/NetSurfP-SS3`](https://huggingface.co/datasets/agemagician/NetSurfP-SS3) | token_classification | MCC |
| `cafa5` | [`andrewdalpino/CAFA5`](https://huggingface.co/datasets/andrewdalpino/CAFA5) | multilabel | F1_Macro |
| `antibiotic_resistance` | [`biomap-research/antibiotic_resistance`](https://huggingface.co/datasets/biomap-research/antibiotic_resistance) | multiclass | AUC |
| `cloning_clf` | [`biomap-research/cloning_clf`](https://huggingface.co/datasets/biomap-research/cloning_clf) | regression | Spearman |
| `enzyme_catalytic_efficiency` | [`biomap-research/enzyme_catalytic_efficiency`](https://huggingface.co/datasets/biomap-research/enzyme_catalytic_efficiency) | regression | Spearman |
| `variant_effect` | [`biomap-research/fitness_prediction`](https://huggingface.co/datasets/biomap-research/fitness_prediction) | regression | Spearman |
| `remote_homology` | [`biomap-research/fold_prediction`](https://huggingface.co/datasets/biomap-research/fold_prediction) | multiclass | Accuracy |
| `material_production` | [`biomap-research/material_production`](https://huggingface.co/datasets/biomap-research/material_production) | binary | AUC |
| `metal_ion_binding` | [`biomap-research/metal_ion_binding`](https://huggingface.co/datasets/biomap-research/metal_ion_binding) | binary | AUC |
| `optimal_ph` | [`biomap-research/optimal_ph`](https://huggingface.co/datasets/biomap-research/optimal_ph) | regression | Spearman |
| `peptide_hla` | [`biomap-research/peptide_HLA_MHC_affinity`](https://huggingface.co/datasets/biomap-research/peptide_HLA_MHC_affinity) | binary | AUC |
| `stability` | [`biomap-research/stability_prediction`](https://huggingface.co/datasets/biomap-research/stability_prediction) | regression | Spearman |
| `temperature_stability` | [`biomap-research/temperature_stability`](https://huggingface.co/datasets/biomap-research/temperature_stability) | multiclass | AUC |
| `fluorescence` | [`cradle-bio/tape-fluorescence`](https://huggingface.co/datasets/cradle-bio/tape-fluorescence) | regression | Spearman |
| `conservation_flip` | `data/conservation_flip` | token_classification | Spearman |
| `disprot` | `data/disprot` | token_classification | MCC |
| `flip2_amylase` | `data/flip2_amylase` | regression | Spearman |
| `flip2_rhomax` | `data/flip2_rhomax` | regression | Spearman |
| `meltome` | [`hazemessam/meltome`](https://huggingface.co/datasets/hazemessam/meltome) | regression | MSE |
| `binary_subcellular_localization` | [`mila-intel/ProtST-BinaryLocalization`](https://huggingface.co/datasets/mila-intel/ProtST-BinaryLocalization) | binary | AUC |
| `subcellular_loc` | [`proteinea/deeploc`](https://huggingface.co/datasets/proteinea/deeploc) | multiclass | AUC |
| `solubility` | [`proteinea/solubility`](https://huggingface.co/datasets/proteinea/solubility) | binary | AUC |
| `scope40_retrieval` | [`tattabio/scope40_test`](https://huggingface.co/datasets/tattabio/scope40_test) | retrieval | Recall@10 |
| `contact_probe` | [`heya5/protein_contact_map`](https://huggingface.co/datasets/heya5/protein_contact_map) | contact_prediction | P@L/5_long |
| `deepet_topt` | [`AI4Protein/DeepET_Topt`](https://huggingface.co/datasets/AI4Protein/DeepET_Topt) | regression | Spearman |
| `ppi_affinity` | [`Synthyra/ppi_affinity`](https://huggingface.co/datasets/Synthyra/ppi_affinity) | regression | Spearman |
| `tcr_pmhc_affinity` | [`GleghornLab/tcr_pmhc_affinity`](https://huggingface.co/datasets/GleghornLab/tcr_pmhc_affinity) | binary | AUC |
| `ss8` | [`GleghornLab/SS8`](https://huggingface.co/datasets/GleghornLab/SS8) | token_classification | F1_Macro |
| `ss3_casp12`, `ss3_casp13`, `ss3_casp14`, `ss3_cb513`, `ss3_ts115`, `ss8_casp12`, `ss8_casp13`, `ss8_casp14`, `ss8_cb513`, `ss8_ts115` | [`proteinea/secondary_structure_prediction`](https://huggingface.co/datasets/proteinea/secondary_structure_prediction) | token_classification | F1_Macro |

Local `data/...` entries are **not** downloaded and are **not** in a fresh
clone. Build them first with the matching script in `scripts/`:
`prep_conservation.py`, `prep_disprot.py`, `prep_flip2.py`. Two of the four
(`conservation_flip`, `disprot`) are in the default preset.

The FLIP downloads (`data.bioembeddings.com`) are slow and time out on
restricted networks. If a machine already has a built copy, copy the directory
instead — it is a plain `datasets.save_to_disk` dump, portable as-is:

```bash
cp -r /path/to/other/ProtBench/data/conservation_flip data/
```

## Verified provenance

Checked against upstream on 2026-05-21. **Read the caveats** — two different
tasks here are both called "disorder" and must not be compared to each other.

| Task | Dataset | Source paper | Caveats |
|---|---|---|---|
| `ss3`, `disorder` | [`agemagician/NetSurfP-SS3`](https://huggingface.co/datasets/agemagician/NetSurfP-SS3) | Klausen et al., *NetSurfP-2.0*, Proteins 2019 ([doi:10.1002/prot.25674](https://doi.org/10.1002/prot.25674)) |  Train/val reshuffled from paper's 10,337/500 to 10,792/646. **CB513 = 511 chains** (vs 513 in paper); **CASP12 = 20** (vs 21). Disorder = PDB-missing-coordinate mask, NOT DisProt / CAID2 — do not cross-compare. |
| `signal_peptide` | [`SaProtHub/Dataset-Signal-Peptides`](https://huggingface.co/datasets/SaProtHub/Dataset-Signal-Peptides) | Teufel et al., *SignalP 6.0*, Nat Biotechnol 2022 ([doi:10.1038/s41587-021-01156-3](https://doi.org/10.1038/s41587-021-01156-3)) | All 25,693 rows packed into HF `train` split; partition is in the `stage` column (20,490 / 2,569 / 2,634).|
| `disprot` | [`LiteFold/DisProt`](https://huggingface.co/datasets/LiteFold/DisProt) | Aspromonte et al., *DisProt 2024*, NAR ([doi:10.1093/nar/gkad928](https://doi.org/10.1093/nar/gkad928)) | Built to local Arrow `data/disprot/` by `scripts/prep_disprot.py`. Per-residue 0/1 = union of curated `region_terms == 'disorder'` spans. Split via DisProt's deterministic `split_bucket` (sha256(id)%10): test=bucket0 (324), val=bucket1 (340), train=buckets2-9 (2,535). **This is the manually-curated CAID-style target — distinct from the NetSurfP `disorder` mask above.** Headline metric MCC (imbalanced, ~17% disordered). |
| `contact_probe` | [`heya5/protein_contact_map`](https://huggingface.co/datasets/heya5/protein_contact_map) | Rao et al., *TAPE*, NeurIPS 2019 ([arXiv:1906.08230](https://arxiv.org/abs/1906.08230)); ProteinNet, AlQuraishi 2019 ([doi:10.1186/s12859-019-2932-0](https://doi.org/10.1186/s12859-019-2932-0)) | TAPE's ProteinNet LMDB converted to parquet: 25,299 train / 224 valid / 40 test (CASP12 targets). `tertiary` is CB coordinates in Ångström, `valid_mask` flags resolved residues (unresolved ones carry `(0,0,0)` and **must** be masked out, or they invent contacts at the origin). Contact = CB–CB < 8 Å. Measured long-range contact rate on the test split: **0.028**, which is the chance floor for `P@L/5_long`. |
| `deepet_topt` | [`AI4Protein/DeepET_Topt`](https://huggingface.co/datasets/AI4Protein/DeepET_Topt) | Li et al., *Learning deep representations of enzyme thermal adaptation*, Protein Science 2022 ([doi:10.1002/pro.4480](https://doi.org/10.1002/pro.4480)); rehosted by VenusFactory, Tan et al. 2025 ([arXiv:2503.15438](https://arxiv.org/abs/2503.15438)) | Optimal growth temperature of the SOURCE ORGANISM in degrees C (range 2-120), **not** the melting temperature of the protein. Do not compare against `thermostability` or `meltome`, which measure protein Tm. 1,478 / 185 / 185. The dataset card cites VenusFactory; the underlying Topt labels are DeepET's, from BRENDA organism OGT. |
| `ppi_affinity` | [`Synthyra/ppi_affinity`](https://huggingface.co/datasets/Synthyra/ppi_affinity) | **Unverified — the card carries no provenance beyond the schema** | Continuous binding affinity (2.4-15.7, consistent with pKd) over sequence pairs. 10,639 / 200 / 200; the `cluster_a` / `cluster_b` columns suggest clustered splits, but by what and at what threshold is not stated. **Test split is only 200 pairs, so single runs are noisy.** Best-effort until traced to an original paper. |
| `tcr_pmhc_affinity` | [`GleghornLab/tcr_pmhc_affinity`](https://huggingface.co/datasets/GleghornLab/tcr_pmhc_affinity) | **Unverified — the card carries no provenance beyond the schema** | 15,041 / 4,485 / 4,485, ~17% binders. `seqs` packs three chains into one string as `CDR3a\|CDR3b\|peptide`; `\|` is out of vocabulary for every tokenizer here, so `patch_unknown_residue_tokens` maps it to `X` and the separator survives as one unknown residue. AUC, not accuracy — always predicting "non-binder" scores 0.83. |
| `ss8` | [`GleghornLab/SS8`](https://huggingface.co/datasets/GleghornLab/SS8) | Klausen et al., *NetSurfP-2.0*, Proteins 2019 ([doi:10.1002/prot.25674](https://doi.org/10.1002/prot.25674)) | 10,792 / 626 / 50. The label column carries 9 symbols: the 8 DSSP states `GHIBESTC` plus `D` for **unassigned**. `D` is **not scored** — it is 6.7% of train and 11.5% of test residues, 91% of them in terminal runs and so trivially predictable from position. Counting it would inflate Accuracy and make F1_Macro a 9-class average that no published Q8 number compares to. Those residues are marked ignore and dropped before fitting, exactly as standard Q8 evaluation masks them. |
| `ss3_*`, `ss8_*` (10 held-out sets) | [`proteinea/secondary_structure_prediction`](https://huggingface.co/datasets/proteinea/secondary_structure_prediction) | Elnaggar et al., *ProtTrans*, TPAMI 2021 ([doi:10.1109/TPAMI.2021.3095381](https://doi.org/10.1109/TPAMI.2021.3095381)); data from NetSurfP-2.0 | Raw CSVs with no HF split config, so each task pins `data_files={"train": "training_hhblits.csv", "test": "<SET>.csv"}`. **Their columns do not match each other** — `CASP13.csv` carries `xyz_coordinates`, `CASP14.csv` also an `Unnamed: 0` index, `training_hhblits.csv` a `cb513_mask` — so each split is loaded separately; one combined `load_dataset` call fails trying to unify the schemas. `dssp3` and `dssp8` are separate columns over the same sequences, so `ss3_cb513` and `ss8_cb513` differ only in label granularity. `dssp8` here emits exactly the 8 DSSP states, matching the `ss8` class ids. |

## Best-effort upstream sources

Not individually re-verified. Treat as a starting point for finding the real
citation, not as the citation itself.

| Task(s) | Upstream | Reference |
|---|---|---|
| `proteingym_*` (8 tasks) | ProteinGym | Notin et al. 2023, doi:10.1101/2023.12.07.570727 |
| `cath_eat` | ProtTucker / EAT | Heinzinger et al. 2022, doi:10.1093/nargab/lqac043 |
| `thermostability`, `aav_flip`, `conservation_flip`, `flip2_*` | FLIP | Dallago et al. 2021, doi:10.1101/2021.11.09.467890 |
| `meltome` | Meltome Atlas, via FLIP | Jarzab et al. 2020, doi:10.1038/s41592-020-0801-4 |
| `beta_lactamase_peer` | PEER | Xu et al. 2022, arXiv:2206.02096 |
| `scope40_retrieval` | SCOPe | Fox et al. 2014, doi:10.1093/nar/gkt1240 |
| `subcellular_loc`, `binary_subcellular_localization` | DeepLoc | Almagro Armenteros et al. 2017, doi:10.1093/bioinformatics/btx431 |
| `fluorescence`, `contact_probe` | TAPE | Rao et al. 2019, arXiv:1906.08230 |
| `go_bp`, `go_cc`, `go_mf` | GO term prediction, DeepFRI-derived splits | Gligorijević et al. 2021, doi:10.1038/s41467-021-23303-9 |
| `ppi_bernett` | Gold-standard PPI splits | Bernett et al. 2024, doi:10.1093/bib/bbae076 |
| `cafa5` | CAFA challenge series | Zhou et al. 2019, doi:10.1186/s13059-019-1835-8 |
| `profet_np_sp_cleaved` | ProFET | Ofer & Linial 2015, doi:10.1093/bioinformatics/btv345 |
| `signalp_binary` | SignalP | Teufel et al. 2022, doi:10.1038/s41587-021-01156-3 |
| `solubility` | DeepSol (per the task name; unconfirmed) | check dataset card |
| `variant_effect` | GB1 fitness (per the task name; unconfirmed) | check dataset card |
| `ec_classification`, `go_mf` | AI4Protein rehost | check dataset card |
| the 11 `biomap-research/*` tasks | third-party rehosts, originals two hops away | check dataset card |
| `rhla_enzyme_mutations` | CAPE | check dataset card |

The `biomap-research/*` block is the largest gap: those are rehosts of other
task collections, and this repo does not record which. If you report one, trace
it yourself.

## Methods and tools

Cite these when you use the corresponding code path.

| Component | Reference |
|---|---|
| MMseqs2 baseline (`mmseqs_baseline.py`) | Steinegger & Söding 2017, doi:10.1038/nbt.3988 |
| phmmer baseline (`hmmer_baseline.py`) | Eddy 2011, doi:10.1371/journal.pcbi.1002195 |
| pyhmmer (phmmer bindings) | Larralde & Zeller 2023, doi:10.1093/bioinformatics/btad214 |
| CATH classification | Sillitoe et al. 2021, doi:10.1093/nar/gkaa1079 |
| Probes and metrics | scikit-learn, Pedregosa et al. 2011 |
| LoRA fine-tuning (`--mode lora`) | Hu et al. 2021, arXiv:2106.09685 |
| Test-time training (`--tta`) | ProteinTTT, arXiv:2411.02109 |
| ProteinGym scoring code (vendored) | Notin et al. 2023 — see the header of `data/proteingym_ref/scoring_utils_proteingym.py` |

## Licensing

Dataset licences vary and are **not** granted by this repo's licence. Check each
HuggingFace card before redistribution or commercial use. Where a licence is
known it is noted in the verified table above; for everything else, assume you
need to check.

`GrimSqueaker/cath43-eat` is CC-BY-4.0, assembled from the GPL-3.0
[Rostlab/EAT](https://github.com/Rostlab/EAT) repository (splits) and
[Zenodo 14675997](https://zenodo.org/records/14675997) (CATH v4.3.0 labels,
CC-BY-4.0). Rebuild with `scripts/build_cath_eat_dataset.py`.

## The CATH midnight-zone task

Accuracy at the homologous-superfamily (H) level, transferring labels from the
69k lookup set to the 150 answerable queries.

**Measured with ProtBench** (1-NN, Euclidean, `-p knn --knn_k 1`):

| Method | Accuracy |
|---|---|
| 3-mer frequencies | 0.0 |
| ESM2-8M | 21.3 |
| ESM2-650M | 42.7 |

**Reported by Heinzinger et al. 2022**, Table 1, on the same splits but with
their own models and embedding pipeline:

| Method | Accuracy |
|---|---|
| Random | 0 |
| MMseqs2 | 35 |
| raw ProtT5 | 64 |
| ProtTucker(ProtT5) | 76 |
| HMMER profiles | 77 |

The two tables are kept apart on purpose: they share splits and scoring but not
models or embedding code, so a row from one is not a like-for-like comparison
against a row from the other. Compare within a table.

The 0.0 for 3-mers is the check that matters. It matches the published random
baseline, which says amino acid composition carries nothing about remote
homology — the task is not leaking a shortcut.

Dataset: [`GrimSqueaker/cath43-eat`](https://huggingface.co/datasets/GrimSqueaker/cath43-eat),
built by `scripts/build_cath_eat_dataset.py`.
