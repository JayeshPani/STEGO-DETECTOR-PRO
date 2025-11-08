import cv2, albumentations as A, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset

class StegoDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.paths = df['filepath'].tolist()
        self.labels = df['label'].astype(np.float32).tolist()
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)  # BGR 3-ch by default
        if img is None: raise FileNotFoundError(self.paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']  # HWC, values already normalized by A.Normalize
        # ensure 3 channels
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        x = torch.from_numpy(img.transpose(2,0,1)).float()
        y = torch.tensor([self.labels[i]], dtype=torch.float32)
        return x, y
