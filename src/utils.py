import torch
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
    @property
    def avg(self):
        return self.sum / self.count if self.count != 0 else 0

@torch.no_grad()
def compute_metrics(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        labels = y.cpu().numpy().ravel()
        all_preds.append(probs)
        all_labels.append(labels)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0
    preds_bin = (all_preds > 0.5).astype(int)
    f1 = f1_score(all_labels, preds_bin, zero_division=0)
    return auc, f1
