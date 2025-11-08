import albumentations as A

IMAGENET_MEAN=(0.5,0.5,0.5)
IMAGENET_STD =(0.5,0.5,0.5)

def train_tfms():
    return A.Compose([
        A.ToGray(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def val_tfms():
    return A.Compose([
        A.ToGray(p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
