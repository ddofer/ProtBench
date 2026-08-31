"""
Benchmark task configurations for Protein Language Models.

Defines the TaskConfig dataclass and the TASKS dictionary: binary, multiclass,
multilabel, regression, retrieval, residue-level (token classification) and
ProteinGym evaluations.

Run `python protein_benchmark_suite.py --list_tasks` for the current inventory.

Usage:
    from benchmark_tasks import TASKS, TaskConfig

    cfg = TASKS["solubility"]
    print(cfg.name, cfg.problem_type)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Metric direction, canonical definition: `_hf_finetune_common` and the
# contrib/proteva reporters import this rather than keeping their own copy.
LOWER_IS_BETTER_METRICS = frozenset({"mse", "mae", "rmse", "loss", "perplexity"})


def metric_greater_is_better(metric_name: str | None) -> bool:
    """True when a LARGER value of ``metric_name`` is the better result."""
    return (metric_name or "").lower() not in LOWER_IS_BETTER_METRICS


def signed_delta(delta: float | None, metric_name: str | None) -> float | None:
    """Re-sign a raw ``candidate - baseline`` delta so POSITIVE always means better."""
    if delta is None:
        return None
    return delta if metric_greater_is_better(metric_name) else -delta


@dataclass
class TaskConfig:
    """Configuration for a single benchmark task."""

    name: str
    dataset: str
    input_map: Dict[str, str]  # Maps internal keys to actual dataset columns
    label_col: str
    problem_type: str  # 'binary', 'multiclass', 'multilabel', 'regression', 'retrieval'
    main_metric: str
    dataset_config: Optional[str] = None
    data_dir: Optional[str] = None  # For datasets that use data_dir (e.g., ProteinGym)
    data_files: Optional[Dict[str, str]] = (
        None  # split -> file, for hub repos with no split config (raw CSV repos)
    )
    train_split: str = "train"
    test_split: str = "test"
    validation_split: Optional[str] = None
    split_column: Optional[str] = (
        None  # Derive train/test rows from a column in one source split
    )
    validation_column_values: Optional[Tuple[str, ...]] = None
    top_k_labels: Optional[int] = None  # For filtering multilabel to top K
    auto_split: bool = False  # If True, split train into train/test (80/20)
    remove_sequence_whitespace: bool = (
        False  # Remove spaces/newlines inside sequence strings
    )
    group_by: Optional[str] = (
        None  # Column to group by for stratified split (e.g., DMS_id)
    )
    label_map: Optional[Dict[str, Any]] = (
        None  # Map raw label values to normalized ones
    )
    eval_mode: str = (
        "standard"  # 'standard', 'proteingym_zeroshot', 'proteingym_supervised'
    )
    bin_col: Optional[str] = (
        None  # Official binary-label column for AUC (ProteinGym DMS_score_bin)
    )
    label_prefix_fields: Optional[int] = (
        None  # Keep only the first N dot-separated fields of each label
        # (SCOPe sccs "a.5.6.1": 2 -> fold "a.5", 3 -> superfamily "a.5.6")
    )

    def __post_init__(self):
        valid_types = {
            "binary",
            "multiclass",
            "multilabel",
            "regression",
            "retrieval",
            "token_classification",
            "contact_prediction",
        }
        if self.problem_type not in valid_types:
            raise ValueError(f"problem_type must be one of {valid_types}")


# Many (not all) of the available benchmark tasks;
FAST_TASKS = [
    "remote_homology",
    # cath_eat: the one fold benchmark whose train/test contamination is controlled
    # (remote_homology and scope40_retrieval are both contaminated).
    "cath_eat",
    "solubility",
    "signalp_binary",
    "profet_np_sp_cleaved",
    "beta_lactamase_peer",
    "peptide_hla",
    "metal_ion_binding",
    "subcellular_loc",
    "ec_classification",
    "variant_effect",
    "fluorescence",
    "stability",
    "enzyme_catalytic_efficiency",
    "ppi_bernett",
    # go_mf excluded: large multilabel task, too slow for the default sweep
    "ss3",
    "conservation_flip",
    "disprot",
]

# Curated low-variance subset for scout comparisons (--very-fast); omits the
# slower/higher-variance FAST_TASKS. Every entry MUST be a key in TASKS.
VERY_FAST_TASKS = [
    "remote_homology",
    "solubility",
    "metal_ion_binding",
    "fluorescence",
    "stability",
    "beta_lactamase_peer",
    "ss3",
    "conservation_flip",
]

# Triage set: "better or worse, and in which domain?" One or two tasks per domain,
# ranked by measured spread. flip2_rhomax / contact_probe / ppi_affinity are OUT --
# n=184/40/200, so their spread is noise and a screen on them cries wolf.
SCREEN_TASKS = [
    "scope40_retrieval",       # structure   n=2,207   no probe fit
    "cath_eat",                # fold        n=219 vs 69k lookup   decontaminated
    "remote_homology",         # fold        n=14,000
    "ec_classification",       # function    n=20,000  most discriminative usable task
    "rhla_enzyme_mutations",   # fitness     n=1,500
    "aav_flip",                # fitness     n=30,000
    "disprot",                 # disorder    n=1,500
    "stability",               # biophysics  n=60,000  priciest here; drop it first
    "conservation_flip",       # biophysics  n=11,000
]

FAST_MAX_SAMPLES = 100_000
# Cap is a sequence count, to keep residue-level probes tractable on CPU.
FAST_TOKEN_CLASS_MAX_SAMPLES = 2_000

_CLINICAL_LABEL_MAP = {"Pathogenic": 1, "Benign": 0, "0": 0, "1": 1}

# ProteinGym variant definitions: (data_dir, label_col, problem_type, main_metric, group_by)
_PROTEINGYM_VARIANTS = {
    "dms_substitutions": (
        "DMS_substitutions",
        "DMS_score",
        "regression",
        "Spearman",
        "DMS_id",
        None,
    ),
    "dms_indels": ("DMS_indels", "DMS_score", "regression", "Spearman", "DMS_id", None),
    "clinical_substitutions": (
        "clinical_substitutions",
        "annotation",
        "binary",
        "AUC",
        "protein_id",
        _CLINICAL_LABEL_MAP,
    ),
    "clinical_indels": (
        "clinical_indels",
        "annotation",
        "binary",
        "AUC",
        "protein_id",
        _CLINICAL_LABEL_MAP,
    ),
}


def _proteingym_tasks(eval_mode: str) -> Dict[str, TaskConfig]:
    """Generate ProteinGym TaskConfig entries for a given eval mode (zeroshot/supervised)."""
    is_zeroshot = eval_mode == "zeroshot"
    tasks = {}
    for variant, (
        data_dir,
        label_col,
        problem_type,
        metric,
        group_by,
        label_map,
    ) in _PROTEINGYM_VARIANTS.items():
        key = f"proteingym_{variant}_{eval_mode}"
        name_parts = " ".join(
            w.upper() if w == "dms" else w.capitalize() for w in variant.split("_")
        )
        mode_label = "Zero-Shot" if is_zeroshot else "Supervised"
        input_map = (
            {"mutant": "mutated_sequence", "wt": "target_seq"}
            if is_zeroshot
            else {"seq": "mutated_sequence"}
        )
        # DMS_score_bin is the official per-assay binary label the leaderboard
        # AUC/MCC use; clinical sets score their annotation label directly.
        bin_col = "DMS_score_bin" if variant.startswith("dms_") else None
        tasks[key] = TaskConfig(
            name=f"ProteinGym {name_parts} ({mode_label})",
            dataset="OATML-Markslab/ProteinGym_v1",
            data_dir=data_dir,
            input_map=input_map,
            label_col=label_col,
            problem_type=problem_type,
            main_metric=metric,
            group_by=group_by,
            label_map=label_map,
            eval_mode=f"proteingym_{eval_mode}",
            bin_col=bin_col,
        )
    return tasks


# Standard held-out SS evaluation sets (ProtTrans / NetSurfP-2.0 distribution);
# all share `training_hhblits.csv` and differ only in the test CSV.
_SS_HELDOUT_SETS = {
    "casp12": "CASP12.csv",
    "casp13": "CASP13.csv",
    "casp14": "CASP14.csv",
    "cb513": "CB513.csv",
    "ts115": "TS115.csv",
}
_SS_HELDOUT_LABELS = {"ss3": "dssp3", "ss8": "dssp8"}
_SS_HELDOUT_DATASET = "proteinea/secondary_structure_prediction"
_SS_HELDOUT_TRAIN_FILE = "training_hhblits.csv"


def _ss_heldout_tasks() -> Dict[str, TaskConfig]:
    """Generate `ss3_casp12` ... `ss8_ts115` (2 label columns x 5 test sets)."""
    tasks = {}
    for states, label_col in _SS_HELDOUT_LABELS.items():
        for set_key, filename in _SS_HELDOUT_SETS.items():
            tasks[f"{states}_{set_key}"] = TaskConfig(
                name=f"Secondary Structure {states[-1]} ({set_key.upper()})",
                dataset=_SS_HELDOUT_DATASET,
                data_files={"train": _SS_HELDOUT_TRAIN_FILE, "test": filename},
                input_map={"seq": "input"},
                label_col=label_col,
                problem_type="token_classification",
                main_metric="F1_Macro",
                test_split="test",
            )
    return tasks


TASKS: Dict[str, TaskConfig] = {
    # Binary Classification
    "ppi_bernett": TaskConfig(
        name="PPI (Bernett Gold Standard)",
        dataset="Synthyra/bernett_gold_ppi",
        input_map={"seq1": "SeqA", "seq2": "SeqB"},
        label_col="labels",
        problem_type="binary",
        main_metric="AUC",
        validation_split="valid",
    ),
    "solubility": TaskConfig(
        name="Solubility (DeepSol)",
        dataset="proteinea/solubility",
        input_map={"seq": "sequences"},
        label_col="labels",
        problem_type="binary",
        main_metric="AUC",
        validation_split="valid",
    ),
    "peptide_hla": TaskConfig(
        name="Peptide-HLA Binding",
        dataset="biomap-research/peptide_HLA_MHC_affinity",
        input_map={"seq": "seq"},
        label_col="label",
        problem_type="binary",
        main_metric="AUC",
        validation_split="valid",
    ),
    "metal_ion_binding": TaskConfig(
        name="Metal Ion Binding",
        dataset="biomap-research/metal_ion_binding",
        input_map={"seq": "seq"},
        label_col="label",
        problem_type="binary",
        main_metric="AUC",
    ),
    "material_production": TaskConfig(
        name="Material Production",
        dataset="biomap-research/material_production",
        input_map={"seq": "seq"},
        label_col="label",
        problem_type="binary",
        main_metric="AUC",
    ),
    "binary_subcellular_localization": TaskConfig(
        name="Binary Subcellular Localization",
        dataset="mila-intel/ProtST-BinaryLocalization",
        input_map={"seq": "prot_seq"},
        label_col="localization",
        problem_type="binary",
        main_metric="AUC",
        train_split="train",
        test_split="test",
        remove_sequence_whitespace=True,
    ),
    "signalp_binary": TaskConfig(
        name="Signal Peptide Prediction (SignalP/ProteinBERT)",
        dataset="GrimSqueaker/SignalP_Binary",
        input_map={"seq": "seq"},
        label_col="label",
        problem_type="binary",
        main_metric="AUC",
        train_split="train",
        test_split="test",
    ),
    "profet_np_sp_cleaved": TaskConfig(
        name="Neuropeptide Precursor Prediction (ProFET/NeuroPID)",
        dataset="GrimSqueaker/ProFET_NP_SP_Cleaved",
        input_map={"seq": "seq"},
        label_col="label",
        problem_type="binary",
        main_metric="AUC",
        train_split="train",
        validation_split="validation",
        test_split="test",
    ),
    # Multi-class Classification
    "remote_homology": TaskConfig(
        name="Remote Homology (Fold)",
        dataset="biomap-research/fold_prediction",
        input_map={"seq": "seq"},
        label_col="label",
        problem_type="multiclass",
        # 1195 imbalanced fold classes: macro-F1 is dominated by singleton folds,
        # so Top-1 Accuracy is the stable, literature-standard headline.
        main_metric="Accuracy",
    ),
    "subcellular_loc": TaskConfig(
        name="Subcellular Localisation",
        dataset="proteinea/deeploc",
        input_map={"seq": "input"},
        label_col="loc",
        problem_type="multiclass",
        main_metric="AUC",
    ),
    "ec_classification": TaskConfig(
        name="EC Classification",
        dataset="AI4Protein/EC",
        input_map={"seq": "aa_seq"},
        label_col="label",
        # EC is MULTILABEL: comma-separated EC numbers per protein. Never declare it
        # multiclass -- each comma string ('130,270') becomes its own class.
        problem_type="multilabel",
        main_metric="F1_Micro",
        validation_split="validation",
    ),
    "antibiotic_resistance": TaskConfig(
        name="Antibiotic Resistance",
        dataset="biomap-research/antibiotic_resistance",
        input_map={"seq": "seq"},
        label_col="label",
        problem_type="multiclass",
        main_metric="AUC",
    ),
    # `seqs` packs three chains as "CDR3a|CDR3b|peptide"; `|` is out of vocabulary,
    # so `patch_unknown_residue_tokens` maps it to X (the intended encoding).
    "tcr_pmhc_affinity": TaskConfig(
        name="TCR-pMHC Binding",
        dataset="GleghornLab/tcr_pmhc_affinity",
        input_map={"seq": "seqs"},
        label_col="labels",
        problem_type="binary",
        main_metric="AUC",
        validation_split="valid",
    ),
    "temperature_stability": TaskConfig(
        name="Temperature Stability",
        dataset="biomap-research/temperature_stability",
        input_map={"seq": "seq"},
        label_col="label",
        problem_type="multiclass",
        main_metric="AUC",
        validation_split="valid",
    ),
    # Multi-label Classification
    "go_mf": TaskConfig(
        name="Molecular Function (GO)",
        dataset="AI4Protein/GO_MF",
        input_map={"seq": "aa_seq"},
        label_col="label",
        problem_type="multilabel",
        main_metric="F1_Macro",
        validation_split="validation",
    ),
    # GO BP/CC: same DeepFRI-derived splits as go_mf; excluded from the presets
    # (thousands of labels each). Request by name.
    "go_bp": TaskConfig(
        name="Biological Process (GO)",
        dataset="AI4Protein/GO_BP",
        input_map={"seq": "aa_seq"},
        label_col="label",
        problem_type="multilabel",
        main_metric="F1_Macro",
        validation_split="validation",
    ),
    "go_cc": TaskConfig(
        name="Cellular Component (GO)",
        dataset="AI4Protein/GO_CC",
        input_map={"seq": "aa_seq"},
        label_col="label",
        problem_type="multilabel",
        main_metric="F1_Macro",
        validation_split="validation",
    ),
    "cafa5": TaskConfig(
        name="CAFA5 (Protein Function)",
        dataset="andrewdalpino/CAFA5",
        dataset_config="mf",
        input_map={"seq": "sequence"},
        label_col="terms",
        problem_type="multilabel",
        main_metric="F1_Macro",
        top_k_labels=500,
    ),
    # Regression
    "variant_effect": TaskConfig(
        name="Variant Effect (GB1)",
        dataset="biomap-research/fitness_prediction",
        input_map={"seq": "seq"},
        label_col="label",
        problem_type="regression",
        main_metric="Spearman",
        validation_split="valid",
    ),
    "fluorescence": TaskConfig(
        name="Fluorescence (TAPE)",
        dataset="cradle-bio/tape-fluorescence",
        input_map={"seq": "primary"},
        label_col="log_fluorescence",
        problem_type="regression",
        main_metric="Spearman",
    ),
    "stability": TaskConfig(
        name="Stability (Biomap)",
        dataset="biomap-research/stability_prediction",
        input_map={"seq": "seq"},
        label_col="label",
        problem_type="regression",
        main_metric="Spearman",
        validation_split="valid",
    ),
    "thermostability": TaskConfig(
        name="Thermostability (FLIP)",
        dataset="SaProtHub/Dataset-Thermostability-FLIP",
        input_map={"seq": "protein"},
        label_col="label",
        problem_type="regression",
        main_metric="Spearman",
        auto_split=True,
    ),
    "meltome": TaskConfig(
        name="Meltome (Melting Temperature Tm)",
        dataset="hazemessam/meltome",
        input_map={"seq": "sequence"},
        label_col="label",
        problem_type="regression",
        main_metric="MSE",
        split_column="split",
        train_split="train",
        test_split="test",
    ),
    # FLIP2 subtasks: local Arrow datasets built by scripts/prep_flip2.py from
    # LiteFold/FLIP2.
    "flip2_amylase": TaskConfig(
        name="Alpha-Amylase Fitness (FLIP2)",
        dataset="data/flip2_amylase",
        input_map={"seq": "sequence"},
        label_col="score",
        problem_type="regression",
        main_metric="Spearman",
    ),
    "flip2_rhomax": TaskConfig(
        name="Rhodopsin Fitness (FLIP2)",
        dataset="data/flip2_rhomax",
        input_map={"seq": "sequence"},
        label_col="score",
        problem_type="regression",
        main_metric="Spearman",
    ),
    # Optimal growth temperature of the source ORGANISM (degrees C), not the
    # protein's melting temperature (`thermostability`, `meltome`).
    "deepet_topt": TaskConfig(
        name="Optimal Growth Temperature (DeepET Topt)",
        dataset="AI4Protein/DeepET_Topt",
        input_map={"seq": "aa_seq"},
        label_col="label",
        problem_type="regression",
        main_metric="Spearman",
        validation_split="validation",
    ),
    # The only pairwise REGRESSION task (`ppi_bernett` is pairwise binary); the
    # seq1/seq2 path concatenates the two pooled embeddings.
    "ppi_affinity": TaskConfig(
        name="PPI Binding Affinity",
        dataset="Synthyra/ppi_affinity",
        input_map={"seq1": "SeqA", "seq2": "SeqB"},
        label_col="labels",
        problem_type="regression",
        main_metric="Spearman",
        validation_split="valid",
    ),
    "optimal_ph": TaskConfig(
        name="Optimal pH",
        dataset="biomap-research/optimal_ph",
        input_map={"seq": "seq"},
        label_col="label",
        problem_type="regression",
        main_metric="Spearman",
    ),
    "enzyme_catalytic_efficiency": TaskConfig(
        name="Enzyme Catalytic Efficiency",
        dataset="biomap-research/enzyme_catalytic_efficiency",
        input_map={"seq": "seq"},
        label_col="label",
        problem_type="regression",
        main_metric="Spearman",
    ),
    "cloning_clf": TaskConfig(
        name="Cloning Classification",
        dataset="biomap-research/cloning_clf",
        input_map={"seq": "seq"},
        label_col="label",
        problem_type="regression",
        main_metric="Spearman",
    ),
    "beta_lactamase_peer": TaskConfig(
        name="beta-lactamase-PEER",
        dataset="SaProtHub/Dataset-Beta_Lactamase-PEER",
        input_map={"seq": "protein"},
        label_col="label",
        problem_type="regression",
        main_metric="Spearman",
        split_column="stage",
        validation_column_values=("valid", "validation", "val"),
        train_split="train",
        test_split="test",
    ),
    "aav_flip": TaskConfig(
        name="AAV Fitness (FLIP)",
        dataset="SaProtHub/Dataset-AAV-FLIP",
        input_map={"seq": "protein"},
        label_col="label",
        problem_type="regression",
        main_metric="Spearman",
        split_column="stage",
        validation_column_values=("valid", "validation", "val"),
        train_split="train",
        test_split="test",
    ),
    "rhla_enzyme_mutations": TaskConfig(
        name="RhlA Enzyme Mutations",
        dataset="SaProtHub/DATASET-CAPE-RhlA-seqlabel",
        input_map={"seq": "protein"},
        label_col="label",
        problem_type="regression",
        main_metric="Spearman",
        split_column="stage",
        validation_column_values=("valid", "validation", "val"),
        train_split="train",
        test_split="test",
    ),
    # Retrieval
    # `family` carries the full SCOPe sccs id ("a.5.6.1"), so this key is
    # FAMILY-level; the two below truncate it to superfamily and fold.
    "scope40_retrieval": TaskConfig(
        name="SCOPe-40 Structural Retrieval",
        dataset="tattabio/scope40_test",
        input_map={"seq": "sequence"},
        label_col="family",
        problem_type="retrieval",
        main_metric="Recall@10",
        train_split="train",
        test_split="train",
    ),
    "scope40_retrieval_superfamily": TaskConfig(
        name="SCOPe-40 Structural Retrieval (superfamily)",
        dataset="tattabio/scope40_test",
        input_map={"seq": "sequence"},
        label_col="family",
        label_prefix_fields=3,
        problem_type="retrieval",
        main_metric="Recall@10",
        train_split="train",
        test_split="train",
    ),
    "scope40_retrieval_fold": TaskConfig(
        name="SCOPe-40 Structural Retrieval (fold)",
        dataset="tattabio/scope40_test",
        input_map={"seq": "sequence"},
        label_col="family",
        label_prefix_fields=2,
        problem_type="retrieval",
        main_metric="Recall@10",
        train_split="train",
        test_split="train",
    ),
    # Midnight-zone homology: queries have no alignment relative in the lookup set
    # (Heinzinger 2022, doi:10.1093/nargab/lqac043; test_h = the 150 answerable
    # queries). Run with `-p knn --knn_k 1` -- that is the paper's method.
    "cath_eat": TaskConfig(
        name="CATH v4.3 Superfamily Transfer (midnight zone)",
        dataset="GrimSqueaker/cath43-eat",
        input_map={"seq": "sequence"},
        label_col="cath_h",
        problem_type="multiclass",
        main_metric="Accuracy",
        train_split="lookup",
        test_split="test_h",
    ),
    # Token Classification (residue-level)
    # Per-residue conservation grades 1-9; local Arrow dataset built by
    # scripts/prep_conservation.py from FLIP FASTA.
    "conservation_flip": TaskConfig(
        name="Residue Conservation (FLIP)",
        dataset="data/conservation_flip",
        input_map={"seq": "sequence"},
        label_col="conservation_labels",
        problem_type="token_classification",
        # Grades are ORDINAL, so nominal macro-F1 misscores near misses; FLIP
        # reports Spearman.
        main_metric="Spearman",
        validation_split="validation",
        test_split="test",
    ),
    "ss3": TaskConfig(
        name="Secondary Structure 3 (NetSurfP-SS3)",
        dataset="agemagician/NetSurfP-SS3",
        input_map={"seq": "input"},
        label_col="label",
        problem_type="token_classification",
        main_metric="F1_Macro",
        validation_split="validation",
        test_split="test",
    ),
    # Labels use a 9-symbol alphabet (8 DSSP states plus `D` for unassigned
    # termini); see `_SS8_ALPHABET` in protein_benchmark_suite.
    "ss8": TaskConfig(
        name="Secondary Structure 8 (DSSP8)",
        dataset="GleghornLab/SS8",
        input_map={"seq": "seqs"},
        label_col="labels",
        problem_type="token_classification",
        main_metric="F1_Macro",
        validation_split="valid",
        test_split="test",
    ),
    "disorder": TaskConfig(
        name="Disorder (NetSurfP-SS3 mask)",
        dataset="agemagician/NetSurfP-SS3",
        input_map={"seq": "input"},
        label_col="disorder",
        problem_type="token_classification",
        main_metric="MCC",
        validation_split="validation",
        test_split="test",
    ),
    "signal_peptide": TaskConfig(
        name="Signal Peptide (SignalP6)",
        dataset="SaProtHub/Dataset-Signal-Peptides",
        input_map={"seq": "protein"},
        label_col="label",
        problem_type="token_classification",
        main_metric="F1_Macro",
        split_column="stage",
        validation_column_values=("valid",),
    ),
    # Curated DisProt region spans (scripts/prep_disprot.py from LiteFold/DisProt),
    # not the NetSurfP `disorder` mask above; MCC is the CAID headline metric.
    "disprot": TaskConfig(
        name="Intrinsic Disorder (DisProt)",
        dataset="data/disprot",
        input_map={"seq": "sequence"},
        label_col="disorder_labels",
        problem_type="token_classification",
        main_metric="MCC",
        validation_split="validation",
        test_split="test",
    ),
    # Contact prediction (pairwise residue-residue)
    # TAPE ProteinNet: `tertiary` holds CB coordinates (Angstrom), contacts are
    # CB-CB < 8A. Coordinates build the LABELS only; the model sees sequence alone.
    "contact_probe": TaskConfig(
        name="Contact Prediction (TAPE ProteinNet)",
        dataset="heya5/protein_contact_map",
        input_map={"seq": "seq"},
        label_col="tertiary",
        problem_type="contact_prediction",
        main_metric="P@L/5_long",
        validation_split="valid",
        test_split="test",
    ),
    # Secondary structure on the standard held-out sets (opt-in by name)
    **_ss_heldout_tasks(),
    # ProteinGym — Zero-Shot (cosine similarity WT vs mutant, per-assay Spearman/AUC)
    **_proteingym_tasks("zeroshot"),
    # ProteinGym — Supervised (intra-assay 80/20 linear probe, per-assay Spearman/AUC)
    **_proteingym_tasks("supervised"),
}

# ProteinGym task keys — large/slow, opt-in only via --proteingym or -t
PROTEINGYM_TASKS = sorted(k for k in TASKS if k.startswith("proteingym_"))

# Retrieval task keys — opt-in only via -t
RETRIEVAL_TASKS = sorted(
    k for k, cfg in TASKS.items() if cfg.problem_type == "retrieval"
)

# Thousands of labels each: excluded from the default sweep, request with --tasks.
MULTILABEL_EXCLUDED_TASKS = frozenset({"go_mf", "go_bp", "go_cc", "cafa5"})

# Ten near-duplicates of `ss3`/`ss8` differing only in the test set; opt-in by
# name, e.g. `--tasks ss8_cb513`.
SS_HELDOUT_TASKS = sorted(_ss_heldout_tasks())

# ~200k rows dominating --no-fast wall-clock for a duplicate thermostability
# signal; opt-in by name.
OVERSIZED_EXCLUDED_TASKS = frozenset({"temperature_stability"})

# Default tasks for --no-fast: every standard probe task minus the opt-in sets.
DEFAULT_TASKS = [
    k
    for k in TASKS
    if k
    not in set(PROTEINGYM_TASKS)
    | set(RETRIEVAL_TASKS)
    | MULTILABEL_EXCLUDED_TASKS
    | set(SS_HELDOUT_TASKS)
    | OVERSIZED_EXCLUDED_TASKS
]


# Display name -> task key: result rows carry the display name, every registry
# lookup is keyed on the short id. Defined here only.
TASK_NAME_TO_KEY = {cfg.name: key for key, cfg in TASKS.items()}
