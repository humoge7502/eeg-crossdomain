import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .evaluate import compute_metrics

class EEGTensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_one_model(model, train_loader, val_loader, device, epochs=100, lr=1e-3, early_stopping_patience=15):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_train_labels = []
    for _, yb in train_loader:
        all_train_labels.extend(yb.numpy().tolist())
    class_counts = np.bincount(all_train_labels, minlength=2)
    class_weights = torch.tensor(
        [len(all_train_labels) / (2.0 * max(c, 1)) for c in class_counts],
        dtype=torch.float32
    ).to(device)
    print(f"[train] Class counts: {class_counts.tolist()}, class weights: {class_weights.cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        print(f"[train] epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"[train] early stopping at epoch {epoch+1}")
                break
    model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_val_loss": best_val_loss}

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        logits = out[0] if isinstance(out, tuple) else out
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob, "metrics": metrics}

def run_lodo_fold(held_out_dataset, all_data, cfg, device):
    from .models import EEGNetSharedEncoder, CrossDomainDecoder
    train_datasets = [k for k in all_data.keys() if k != held_out_dataset]
    X_train = np.concatenate([all_data[k][0] for k in train_datasets], axis=0)
    y_train = np.concatenate([all_data[k][1] for k in train_datasets], axis=0)
    X_test, y_test = all_data[held_out_dataset]
    n_channels = X_train.shape[2]
    n_timepoints = X_train.shape[3]
    encoder = EEGNetSharedEncoder(n_channels=n_channels, n_timepoints=n_timepoints, n_filters_temporal=cfg["model"]["n_filters_temporal"], n_filters_spatial=cfg["model"]["n_filters_spatial"], embedding_dim=cfg["model"]["embedding_dim"], dropout=cfg["model"]["dropout"])
    model = CrossDomainDecoder(encoder, embedding_dim=cfg["model"]["embedding_dim"], n_classes=2, dropout=cfg["model"]["dropout"])
    n_val = max(1, int(0.15 * len(y_train)))
    perm = np.random.default_rng(cfg["training"]["seed"]).permutation(len(y_train))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_loader = DataLoader(EEGTensorDataset(X_train[train_idx], y_train[train_idx]), batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(EEGTensorDataset(X_train[val_idx], y_train[val_idx]), batch_size=cfg["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(EEGTensorDataset(X_test, y_test), batch_size=cfg["training"]["batch_size"], shuffle=False)
    trained = train_one_model(model, train_loader, val_loader, device, epochs=cfg["training"]["epochs"], lr=cfg["training"]["learning_rate"], early_stopping_patience=cfg["training"]["early_stopping_patience"])
    test_results = evaluate_model(trained["model"], test_loader, device)
    return {"held_out": held_out_dataset, "trained_on": train_datasets, "metrics": test_results["metrics"], "history": trained["history"]}
