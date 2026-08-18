"""EEGNet & DeepConvNet (Lawhern 2018; Schirrmeister 2017), input [B,1,C,T]. Small, seedable, no external deps."""
import torch, torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, n_ch=7, n_t=250, F1=8, D=2, F2=16, k1=64, drop=0.25, n_classes=2):
        super().__init__(); k1 = min(k1, n_t)
        self.b1 = nn.Sequential(nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False), nn.BatchNorm2d(F1),
                                nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.b2 = nn.Sequential(nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False), nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad(): n = self.b2(self.b1(torch.zeros(1, 1, n_ch, n_t))).numel()
        self.fc = nn.Linear(n, n_classes)
    def forward(self, x): return self.fc(self.b2(self.b1(x)).flatten(1))

class DeepConvNet(nn.Module):
    def __init__(self, n_ch=7, n_t=250, n_classes=2, drop=0.5, f=(25, 50, 100, 200)):
        super().__init__(); L = [nn.Conv2d(1, f[0], (1, 5)), nn.Conv2d(f[0], f[0], (n_ch, 1), bias=False), nn.BatchNorm2d(f[0]), nn.ELU(), nn.MaxPool2d((1, 2)), nn.Dropout(drop)]
        for i in range(1, len(f)): L += [nn.Conv2d(f[i - 1], f[i], (1, 5), bias=False), nn.BatchNorm2d(f[i]), nn.ELU(), nn.MaxPool2d((1, 2)), nn.Dropout(drop)]
        self.body = nn.Sequential(*L)
        with torch.no_grad(): n = self.body(torch.zeros(1, 1, n_ch, n_t)).numel()
        self.fc = nn.Linear(n, n_classes)
    def forward(self, x): return self.fc(self.body(x).flatten(1))
