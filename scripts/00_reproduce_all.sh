#!/bin/bash
# =============================================================================
# Master reproduction script for the cross-domain EEG transfer project.
# Runs the full pipeline in dependency order, with checkpointing (skips a
# step if its expected output already exists).
#
# IMPORTANT SCOPE NOTE: this script assumes data/processed/<name>/<name>_epochs_common.npz
# already exists on disk (from original project setup, predates this session's
# work -- provenance not fully re-derivable at the time this script was written).
# It DOES fully regenerate data/processed_v2/ from raw sources.
#
# THREE DISTINCT NORMALIZATION STRATEGIES are used in this pipeline, each
# independently verified for its own architecture/data path -- this is
# intentional, not an inconsistency to "fix":
#   A. Per-subject normalization (02b) -> data/processed_v2/*_features_persubj_v2.npz
#      Used by: tabular Option A-D (06-09)
#   B. Per-dataset global z-scoring, in-memory only, inside 07_diagnose_and_retry_lodo.py
#      Used by: deep-model (EEGNet) LODO -- the "scaled" results in the log
#   C. Per-trial per-channel z-scoring, inside scripts 13/14 (added this session)
#      Used by: rigorous permutation tests, positive control (DeepConvNet)
#
# Usage: bash scripts/00_reproduce_all.sh [--gpu N]
# =============================================================================
set -e  # exit on first error
cd "$(dirname "$0")/.."

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPU: $GPU"
echo "Working directory: $(pwd)"

step() {
    echo ""
    echo "============================================================"
    echo "STEP: $1"
    echo "============================================================"
}

skip_if_exists() {
    if [ -f "$1" ]; then
        echo "[skip] $1 already exists"
        return 0
    fi
    return 1
}

# --- Stage 1: Preprocessing & normalization fix ---
step "1. Preprocess and harmonize (data/processed_v2/)"
if ! skip_if_exists "data/processed_v2/neuma/neuma_features_persubj_v2.npz"; then
    python scripts/02_preprocess_and_harmonize.py
fi

# --- Stage 2: Tabular Option A-D (uses corrected per-subject normalization) ---
step "2. Option A (full-duration features + CORAL)"
if ! skip_if_exists "results/option_A_results.json"; then
    python scripts/06_option_A_tabular_lodo.py
fi

step "2b. Option A equivalence audit"
if ! skip_if_exists "results/option_A_equivalence_report.json"; then
    python scripts/12_option_A_equivalence.py
fi

step "3. Option C (subject-wise multitask, validated)"
if ! skip_if_exists "results/option_C_subjectwise_results.json"; then
    python scripts/08_option_C_multitask_subjectwise.py
fi

# --- Stage 3: Deep-model LODO (EEGNet, per-dataset z-scoring, trusted "scaled" results) ---
step "4. Deep-model LODO (diagnose + retry, produces the 'scaled' trusted results)"
if ! skip_if_exists "results/lodo_scaled_permutation_equivalence_report.md"; then
    python scripts/07_diagnose_and_retry_lodo.py
fi

# --- Stage 4: Rigorous permutation significance tests (all 3 datasets) ---
step "5. Rigorous permutation tests (N=500, within-subject, Section-17 compliant)"
for ds in neuma restaurant_logo ds007406; do
    if ! skip_if_exists "results/${ds}_deepconvnet_permutation_test_RIGOROUS.json"; then
        python scripts/13_permutation_test_rigorous.py --dataset $ds --n-permutations 500
    fi
done

# --- Stage 5: Positive control ---
step "6. Positive control (subject-ID decoding)"
if ! skip_if_exists "results/neuma_positive_control_subject_id.json"; then
    python scripts/14_positive_control_subject_id.py --dataset neuma
fi

# --- Stage 6: Representation similarity ---
step "7. CKA with null/positive-control baselines (methodological finding: unreliable for small N)"
if ! skip_if_exists "results/cka_with_controls_results.json"; then
    python scripts/16_cka_with_controls.py
fi

step "8. Principal-angle analysis (primary representation-similarity method)"
if ! skip_if_exists "results/principal_angle_analysis_results.json"; then
    python scripts/17_principal_angle_analysis.py
fi

step "9. Principal-angle significance test (bootstrap CI)"
if ! skip_if_exists "results/principal_angle_significance_neuma_restlogo.json"; then
    python scripts/18_principal_angle_significance.py
fi

# --- Stage 7: Multi-seed robustness ---
step "10. Multi-seed robustness: representation alignment"
if ! skip_if_exists "results/multiseed_principal_angle_results.json"; then
    python scripts/19_multiseed_principal_angle.py
fi

step "11. Multi-seed robustness: Option A"
if ! skip_if_exists "results/multiseed_option_A_results.json"; then
    python scripts/20_multiseed_option_A.py
fi

echo ""
echo "============================================================"
echo "REPRODUCTION COMPLETE"
echo "============================================================"
echo "See results/EXPERIMENT_LOG.md (Sections 18-24) for the narrative"
echo "interpretation of all results produced by this script."

# --- Stage 8: Architecture generality tests (Options B & C) ---
step "12. MDM Riemannian geometry classifier (Option B)"
for ds in neuma restaurant_logo ds007406; do
    if ! skip_if_exists "results/mdm_${ds}_results.json"; then
        python scripts/21_mdm_riemannian_lodo.py --dataset $ds --n-permutations 1000
    fi
done

step "13. CBraMod pretrained foundation model (Option C)"
if ! skip_if_exists "results/cbramod_permutation_results.json"; then
    python scripts/22_cbramod_frozen_features.py
    python scripts/23_cbramod_permutation_test.py
fi

step "14. MDM-ds007406 robustness check (LOSO + permutation-seed stability)"
if ! skip_if_exists "results/mdm_ds007406_robustness.json"; then
    python scripts/24_mdm_ds007406_robustness.py
fi
