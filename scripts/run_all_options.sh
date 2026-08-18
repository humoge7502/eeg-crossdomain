#!/usr/bin/env bash
cd ~/24PHD1314/eeg-crossdomain-workspace
source venv/bin/activate

echo "=== Starting Option A ===" 
python3 scripts/06_option_A_tabular_lodo.py > option_A_log.txt 2>&1
echo "=== Option A done, starting Option B ===" 
python3 scripts/07_option_B_engagement_relabel.py > option_B_log.txt 2>&1
echo "=== Option B done, starting Option C ===" 
python3 scripts/08_option_C_multitask.py > option_C_log.txt 2>&1
echo "=== Option C done, starting Option D ===" 
python3 scripts/09_option_D_dann.py > option_D_log.txt 2>&1
echo "=== Option D done. Compiling final summary ===" 

python3 -c "
import json
from pathlib import Path

lines = ['', '## 11. Case Study: Options A-D Comparison', '']
lines.append('| Option | Description | Key metric | Result |')
lines.append('|---|---|---|---|')

try:
    with open('results/option_A_results.json') as f:
        a = json.load(f)
    k1 = a['no_alignment']['summary']['cohen_kappa']['mean']
    k2 = a['coral_alignment']['summary']['cohen_kappa']['mean']
    lines.append(f'| A: Full-duration tabular features | No truncation, band-power features | LODO kappa | no-align={k1:.4f}, CORAL={k2:.4f} |')
except Exception as e:
    lines.append(f'| A | ERROR: {e} | | |')

try:
    with open('results/option_B_results.json') as f:
        b = json.load(f)
    k = b['summary']['cohen_kappa']['mean']
    lines.append(f'| B: Feature-derived engagement relabel | Task-agnostic arousal-based label (median split beta+gamma minus alpha) | LODO kappa | {k:.4f} |')
except Exception as e:
    lines.append(f'| B | ERROR: {e} | | |')

try:
    with open('results/option_C_results.json') as f:
        c = json.load(f)
    deltas = {name: c['multitask'][name]['cohen_kappa'] - c['solo'][name]['cohen_kappa'] for name in c['multitask']}
    delta_str = ', '.join(f'{k}:{v:+.4f}' for k,v in deltas.items())
    lines.append(f'| C: Multi-task shared encoder | Joint pretraining vs solo per-dataset | kappa delta (multitask-solo) | {delta_str} |')
except Exception as e:
    lines.append(f'| C | ERROR: {e} | | |')

try:
    with open('results/option_D_results.json') as f:
        d = json.load(f)
    k = d['summary']['cohen_kappa']['mean']
    lines.append(f'| D: DANN (domain-adversarial) | Gradient-reversal domain-invariant training | LODO kappa | {k:.4f} |')
except Exception as e:
    lines.append(f'| D | ERROR: {e} | | |')

lines.append('')
lines.append('Baseline for comparison: naive zero-shot LODO (Experiment 4) kappa = -0.0003 +/- 0.0004')
lines.append('')

with open('results/EXPERIMENT_LOG.md', 'a') as f:
    f.write('\n'.join(lines) + '\n')

print('Combined summary appended to results/EXPERIMENT_LOG.md')
"

echo "=== ALL OPTIONS COMPLETE ==="
