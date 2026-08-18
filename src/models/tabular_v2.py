"""Shared tabular encoder + heads used for MLP baseline (Exp2) and transfer (Exp4)."""
import torch, torch.nn as nn
class TabularEncoder(nn.Module):
    def __init__(self, in_dim=36, hidden=64, emb=32, drop=0.2):
        super().__init__(); self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ELU(), nn.Dropout(drop), nn.Linear(hidden, emb), nn.ELU()); self.emb_dim = emb
    def forward(self, x): return self.net(x)
class Head(nn.Module):
    def __init__(self, emb=32, n_classes=2): super().__init__(); self.fc = nn.Linear(emb, n_classes)
    def forward(self, z): return self.fc(z)
class EncHead(nn.Module):
    def __init__(self, enc, head): super().__init__(); self.enc, self.head = enc, head
    def forward(self, x): return self.head(self.enc(x))
