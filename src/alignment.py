import numpy as np

def compute_covariances(epochs_data):
    from pyriemann.estimation import Covariances
    cov_estimator = Covariances(estimator="oas")
    return cov_estimator.fit_transform(epochs_data)

def riemannian_align(covariances_by_dataset):
    from pyriemann.utils.mean import mean_riemann
    from scipy.linalg import fractional_matrix_power
    aligned = {}
    for dataset_name, covs in covariances_by_dataset.items():
        ref = mean_riemann(covs)
        ref_inv_sqrt = fractional_matrix_power(ref, -0.5).real
        aligned_covs = np.array([ref_inv_sqrt @ c @ ref_inv_sqrt.T for c in covs])
        aligned[dataset_name] = aligned_covs
        print(f"[alignment] Riemannian-aligned {dataset_name}: {covs.shape[0]} epochs")
    return aligned

def coral_align(source_features, target_features):
    eps = 1e-5
    d = source_features.shape[1]
    cov_source = np.cov(source_features, rowvar=False) + eps * np.eye(d)
    cov_target = np.cov(target_features, rowvar=False) + eps * np.eye(d)
    src_mean = source_features.mean(axis=0)
    centered = source_features - src_mean
    from scipy.linalg import fractional_matrix_power
    whitening = fractional_matrix_power(cov_source, -0.5).real
    coloring = fractional_matrix_power(cov_target, 0.5).real
    aligned = centered @ whitening @ coloring
    aligned += target_features.mean(axis=0)
    return aligned

def apply_alignment(covariances_by_dataset, method):
    if method == "riemannian":
        return riemannian_align(covariances_by_dataset)
    elif method == "none":
        return covariances_by_dataset
    elif method == "coral":
        raise NotImplementedError("call coral_align() directly on Stage-2 features instead")
    else:
        raise ValueError(f"Unknown alignment method: {method}")
