import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold

class DeepConvNet(nn.Module):
    def __init__(self, n_channels, n_timepoints, n_classes=2, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10)),
            nn.Conv2d(25, 25, kernel_size=(n_channels, 1)),
            nn.BatchNorm2d(25), nn.ELU(), nn.MaxPool2d((1, 3)), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 10)),
            nn.BatchNorm2d(50), nn.ELU(), nn.MaxPool2d((1, 3)), nn.Dropout(dropout))
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 10)),
            nn.BatchNorm2d(100), nn.ELU(), nn.MaxPool2d((1, 3)), nn.Dropout(dropout))
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            flat_size = self._forward_features(dummy).shape[1]
        self.classifier = nn.Linear(flat_size, n_classes)

    def _forward_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.flatten(start_dim=1)

    def forward(self, x):
        feats = self._forward_features(x)
        return self.classifier(feats), feats


def svm_psd_pai_baseline(features, labels, groups, cv_folds=5):
    """
    Subject-wise cross-validation: no subject's epochs appear in both
    train and test within a fold. Falls back to fewer folds if a
    dataset has fewer unique subjects than cv_folds requires.
    """
    n_groups = len(np.unique(groups))
    cv_folds = min(cv_folds, n_groups)
    if cv_folds < 2:
        return {"model": "SVM-RBF (PSD+PAI features)", "error": "too few subjects for group CV"}

    clf = SVC(kernel="rbf", gamma="scale")
    sgkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in sgkf.split(features, labels, groups=groups):
        clf.fit(features[train_idx], labels[train_idx])
        scores.append(clf.score(features[test_idx], labels[test_idx]))
    scores = np.array(scores)
    return {
        "model": "SVM-RBF (PSD+PAI features)",
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std": float(scores.std()),
        "cv_folds": cv_folds,
        "split_type": "subject-wise (StratifiedGroupKFold)",
    }


def mdm_riemannian_baseline(epochs_data, labels, groups, cv_folds=5):
    from pyriemann.estimation import Covariances
    from pyriemann.classification import MDM
    from sklearn.pipeline import make_pipeline

    n_groups = len(np.unique(groups))
    cv_folds = min(cv_folds, n_groups)
    if cv_folds < 2:
        return {"model": "MDM (Riemannian covariance)", "error": "too few subjects for group CV"}

    pipeline = make_pipeline(Covariances(estimator="oas"), MDM())
    sgkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in sgkf.split(epochs_data, labels, groups=groups):
        pipeline.fit(epochs_data[train_idx], labels[train_idx])
        scores.append(pipeline.score(epochs_data[test_idx], labels[test_idx]))
    scores = np.array(scores)
    return {
        "model": "MDM (Riemannian covariance)",
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std": float(scores.std()),
        "cv_folds": cv_folds,
        "split_type": "subject-wise (StratifiedGroupKFold)",
    }
