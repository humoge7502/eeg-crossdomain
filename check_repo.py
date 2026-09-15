import subprocess, os

REPO_ROOT = "."

for root, dirs, files in os.walk(REPO_ROOT):
    dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '.ipynb_checkpoints', 'node_modules')]
    for f in files:
        if f.endswith(('.py', '.ipynb')) and any(k in f.lower() for k in 
            ['sweep', 'lodo', 'normal', 'model', 'train', 'option', 'harmoniz']):
            print(os.path.join(root, f))
