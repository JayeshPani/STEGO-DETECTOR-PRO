import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def build_model(pretrained: bool = True):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = efficientnet_b0(weights=weights)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, 1)
    return m
