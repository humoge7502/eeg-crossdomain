import torch
import torch.nn as nn

class EEGNetSharedEncoder(nn.Module):
    def __init__(self, n_channels, n_timepoints, n_filters_temporal=8, n_filters_spatial=16, embedding_dim=64, dropout=0.25):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, n_filters_temporal, kernel_size=(1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters_temporal)
        self.spatial_conv = nn.Conv2d(n_filters_temporal, n_filters_spatial, kernel_size=(n_channels, 1), groups=n_filters_temporal, bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters_spatial)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        self.sep_conv = nn.Conv2d(n_filters_spatial, n_filters_spatial, kernel_size=(1, 16), padding=(0, 8), groups=n_filters_spatial, bias=False)
        self.bn3 = nn.BatchNorm2d(n_filters_spatial)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            flat_size = self._forward_features(dummy).shape[1]
        self.embedding = nn.Linear(flat_size, embedding_dim)

    def _forward_features(self, x):
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.sep_conv(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        return x.flatten(start_dim=1)

    def forward(self, x):
        feats = self._forward_features(x)
        return self.embedding(feats)

class TabularMLPEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=64, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, embedding_dim), nn.ReLU())

    def forward(self, x):
        return self.net(x)

class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim=64, n_classes=2, dropout=0.25):
        super().__init__()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, n_classes))

    def forward(self, x):
        return self.head(x)

class CrossDomainDecoder(nn.Module):
    def __init__(self, encoder, embedding_dim=64, n_classes=2, dropout=0.25):
        super().__init__()
        self.encoder = encoder
        self.head = ClassificationHead(embedding_dim, n_classes, dropout)

    def forward(self, x):
        emb = self.encoder(x)
        logits = self.head(emb)
        return logits, emb
