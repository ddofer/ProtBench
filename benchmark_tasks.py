"""
Benchmark task configurations for Protein Language Models.

Defines the TaskConfig dataclass and the TASKS dictionary with 35 benchmark tasks
including binary/multiclass/multilabel classification, regression, retrieval,
and ProteinGym evaluations.

Usage:
    from benchmark_tasks import TASKS, TaskConfig

    cfg = TASKS["solubility"]
    print(cfg.name, cfg.problem_type)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


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

    def __post_init__(self):
        valid_types = {
            "binary",
            "multiclass",
            "multilabel",
            "regression",
            "retrieval",
            "token_classification",
        }
        if self.problem_type not in valid_types:
            raise ValueError(f"problem_type must be one of {valid_types}")


# Many (not all) of the available benchmark tasks;
FAST_TASKS = [
    "remote_homology",
    "solubility",
    "signalp_binary",
    "profet_np_sp_cleaved",
    "beta_lactamase_peer",
    "peptide_hla",
    "metal_ion_binding",
    "subcellular_loc",
    # "binary_subcellular_localization",
    "ec_classification",
    "variant_effect",
    "fluorescence",
    "stability",
    # "thermostability",
    "enzyme_catalytic_efficiency",
    # "antibiotic_resistance",
    "ppi_bernett",
    # go_mf excluded: multilabel task, not counted in paper wins
    "chezod_disorder",
]

FAST_MAX_SAMPLES = 100_000

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
        # Preserve original display names: "DMS Substitutions", "Zero-Shot", etc.
        name_parts = " ".join(
            w.upper() if w == "dms" else w.capitalize() for w in variant.split("_")
        )
        mode_label = "Zero-Shot" if is_zeroshot else "Supervised"
        input_map = (
            {"mutant": "mutated_sequence", "wt": "target_seq"}
            if is_zeroshot
            else {"seq": "mutated_sequence"}
        )
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
        )
    return tasks


TASKS: Dict[str, TaskConfig] = {
    # =========================================================================
    # Binary Classification
    # =========================================================================
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
    # =========================================================================
    # Multi-class Classification
    # =========================================================================
    "remote_homology": TaskConfig(
        name="Remote Homology (Fold)",
        dataset="biomap-research/fold_prediction",
        input_map={"seq": "seq"},
        label_col="label",
        problem_type="multiclass",
        # Large multiclass (1195 fold classes); sklearn roc_auc_score is not
        # run for this task so AUC is never populated — F1_Macro is the
        # actual primary metric used across all benchmark runs.
        main_metric="F1_Macro",
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
        problem_type="multiclass",
        # Large multiclass (hundreds of EC classes); AUC is never computed
        # by the benchmark suite — F1_Macro is the actual primary metric.
        main_metric="F1_Macro",
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
    "temperature_stability": TaskConfig(
        name="Temperature Stability",
        dataset="biomap-research/temperature_stability",
        input_map={"seq": "seq"},
        label_col="label",
        problem_type="multiclass",
        main_metric="AUC",
        validation_split="valid",
    ),
    # =========================================================================
    # Multi-label Classification
    # =========================================================================
    "go_mf": TaskConfig(
        name="Molecular Function (GO)",
        dataset="AI4Protein/GO_MF",
        input_map={"seq": "aa_seq"},
        label_col="label",
        problem_type="multilabel",
        main_metric="F1_Macro",
        validation_split="validation",
        # top_k_labels=300,
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
    # =========================================================================
    # Regression
    # =========================================================================
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
    "chezod_disorder": TaskConfig(
        name="CheZoD Disorder (Mean Z-Score)",
        dataset="data/chezod",
        input_map={"seq": "sequence"},
        label_col="disorder_mean",
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
    # =========================================================================
    # Retrieval
    # =========================================================================
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
    ## disable std task for now
    # "chezod_disorder_std": TaskConfig(
    #     name="CheZoD Disorder (Std Z-Score)",
    #     dataset="data/chezod",
    #     input_map={"seq": "sequence"},
    #     label_col="disorder_std",
    #     problem_type="regression",
    #     main_metric="Spearman",
    # ),
    # =========================================================================
    # Token Classification (residue-level)
    # =========================================================================
    "ss3": TaskConfig(
        name="Secondary Structure 3 (NetSurfP-SS3)",
        dataset="agemagician/NetSurfP-SS3",
        input_map={"seq": "input"},
        label_col="label",
        problem_type="token_classification",
        main_metric="Accuracy",
        validation_split="validation",
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
    # =========================================================================
    # ProteinGym — Zero-Shot (cosine similarity WT vs mutant, per-assay Spearman/AUC)
    # =========================================================================
    **_proteingym_tasks("zeroshot"),
    # =========================================================================
    # ProteinGym — Supervised (intra-assay 80/20 linear probe, per-assay Spearman/AUC)
    # =========================================================================
    **_proteingym_tasks("supervised"),
}

# ProteinGym task keys — large/slow, opt-in only via --proteingym or -t
PROTEINGYM_TASKS = sorted(k for k in TASKS if k.startswith("proteingym_"))

# Retrieval task keys — opt-in only via -t
RETRIEVAL_TASKS = sorted(
    k for k, cfg in TASKS.items() if cfg.problem_type == "retrieval"
)

# Multilabel tasks excluded from default wins/benchmarks per paper policy
# (results are kept in TASKS for historical compatibility, but these are
# not counted in model comparisons and not run by default).
MULTILABEL_EXCLUDED_TASKS = frozenset({"go_mf", "cafa5"})

# Default tasks for --no-fast: all standard probe tasks, excluding ProteinGym,
# opt-in retrieval tasks, and multilabel-excluded tasks.
DEFAULT_TASKS = [
    k
    for k in TASKS
    if k not in set(PROTEINGYM_TASKS) | set(RETRIEVAL_TASKS) | MULTILABEL_EXCLUDED_TASKS
]
