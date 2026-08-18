#!/usr/bin/env bash
cd ~/24PHD1314/eeg-crossdomain-workspace
source venv/bin/activate

echo "=== [A] Re-running full preprocessing (adds subject_ids to raw epochs) ==="
python3 scripts/02_preprocess_all.py > preprocess_log3.txt 2>&1
echo "=== [A] Done ==="

echo "=== [B] Re-running resample/truncate (carries subject_ids through) ==="
python3 scripts/02b_resample_truncate_epochs.py > resample_log2.txt 2>&1
echo "=== [B] Done ==="

echo "=== [C] Re-running within-dataset baselines with SUBJECT-WISE CV ==="
python3 scripts/03_run_within_dataset_baselines.py > baselines_subjectwise_log.txt 2>&1
echo "=== [C] Done ==="

echo "=== Compiling comparison: epoch-wise (leaky) vs subject-wise (correct) ==="
python3 -c "
import json
lines = ['', '## 12. CRITICAL FIX: Subject-wise (leakage-free) re-validation', '']
lines.append('Following external review, the original within-dataset CV (Section 7) split by')
lines.append('EPOCH, not SUBJECT — meaning the same person could appear in both train and test')
lines.append('folds, inflating reported kappa via subject-identity leakage rather than genuine')
lines.append('task-signal decoding. All baselines were re-run using StratifiedGroupKFold, grouped')
lines.append('by real subject ID, so no subject ever appears in both train and test within a fold.')
lines.append('')
lines.append('| Dataset | Model | Kappa (epoch-wise, ORIGINAL) | Kappa (subject-wise, CORRECTED) |')
lines.append('|---|---|---|---|')

try:
    with open('results/within_dataset_baselines_subjectwise.json') as f:
        new = json.load(f)
    old_kappa = {
        ('neuma','eegnet'): 0.0599, ('neuma','deepconvnet'): 0.0610,
        ('restaurant_logo','eegnet'): 0.0106, ('restaurant_logo','deepconvnet'): 0.0000,
        ('ds007406','eegnet'): 0.0000, ('ds007406','deepconvnet'): 0.0000,
    }
    for dataset in ['neuma','restaurant_logo','ds007406']:
        for model in ['eegnet','deepconvnet']:
            if model in new.get(dataset, {}) and 'mean' in new[dataset][model]:
                new_k = new[dataset][model]['mean'].get('cohen_kappa', float('nan'))
                old_k = old_kappa.get((dataset, model), float('nan'))
                lines.append(f'| {dataset} | {model} | {old_k:.4f} | {new_k:.4f} |')
            else:
                lines.append(f'| {dataset} | {model} | ERROR/SKIPPED | see baselines_subjectwise_log.txt |')
except Exception as e:
    lines.append(f'| ERROR loading results: {e} | | | |')

lines.append('')
with open('results/EXPERIMENT_LOG.md', 'a') as f:
    f.write(chr(10).join(lines) + chr(10))
print('Subject-wise comparison appended to EXPERIMENT_LOG.md')
"
echo "=== ALL SUBJECT-WISE FIXES COMPLETE ==="
