#!/usr/bin/env bash
cd ~/24PHD1314/eeg-crossdomain-workspace
source venv/bin/activate

echo "=== [1] Option C, subject-wise ==="
python3 scripts/08_option_C_multitask_subjectwise.py > option_C_subjectwise_log.txt 2>&1
echo "=== [1] Done ==="

echo "=== [2] Permutation test: NeuMa DeepConvNet (200 permutations, this is the slow part) ==="
python3 scripts/10_permutation_test_neuma_deepconvnet.py > permutation_test_log.txt 2>&1
echo "=== [2] Done ==="

echo "=== Appending final results to EXPERIMENT_LOG.md ==="
python3 -c "
import json
lines = ['', '## 13. Option C re-validated (subject-wise) + Permutation Test', '']

try:
    with open('results/option_C_subjectwise_results.json') as f:
        c = json.load(f)
    lines.append('### Option C, subject-wise train/test split (corrected)')
    lines.append('')
    lines.append('| Dataset | Multitask kappa | Solo kappa | Delta |')
    lines.append('|---|---|---|---|')
    for name in c['multitask']:
        mt = c['multitask'][name]['cohen_kappa']
        so = c['solo'][name]['cohen_kappa']
        lines.append(f'| {name} | {mt:.4f} | {so:.4f} | {mt-so:+.4f} |')
    lines.append('')
except Exception as e:
    lines.append(f'Option C subject-wise: ERROR {e}')

try:
    with open('results/neuma_deepconvnet_permutation_test.json') as f:
        pt = json.load(f)
    lines.append('### Permutation significance test: NeuMa DeepConvNet (the primary validated result)')
    lines.append('')
    lines.append(f\"Observed kappa: {pt['observed_kappa']:.4f}\")
    lines.append(f\"Null distribution (200 label-shuffled permutations, subject-wise CV each): mean={pt['null_mean']:.4f}, std={pt['null_std']:.4f}\")
    lines.append(f\"Null 95% range: [{pt['null_95ci'][0]:.4f}, {pt['null_95ci'][1]:.4f}]\")
    lines.append(f\"p-value: {pt['p_value']:.4f}\")
    lines.append(f\"Significant at 0.05: {pt['significant_at_0.05']}\")
    lines.append('')
except Exception as e:
    lines.append(f'Permutation test: ERROR {e}')

with open('results/EXPERIMENT_LOG.md', 'a') as f:
    f.write(chr(10).join(lines) + chr(10))
print('Appended to EXPERIMENT_LOG.md')
"
echo "=== OPTION C + PERMUTATION TEST COMPLETE ==="
