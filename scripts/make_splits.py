import argparse, os, glob, pandas as pd
from sklearn.model_selection import train_test_split

def main(root='data/raw/alaska2', out_dir='data/splits', test_size=0.1, val_size=0.1, seed=42):
    # Heuristic: look for file/dir names containing 'Cover' and 'Stego'
    cover = sorted(glob.glob(os.path.join(root, '**/*Cover*.*'), recursive=True))
    stego = sorted(glob.glob(os.path.join(root, '**/*Stego*.*'), recursive=True))
    data = [(p, 0) for p in cover] + [(p, 1) for p in stego]
    df = pd.DataFrame(data, columns=['filepath','label'])
    if df.empty:
        raise RuntimeError(f'No files found under {root}. Adjust patterns in make_splits.py for ALASKA2 layout.')
    train_df, tmp_df = train_test_split(df, test_size=(test_size+val_size), stratify=df['label'], random_state=seed)
    val_rel = val_size/(test_size+val_size)
    val_df, test_df = train_test_split(tmp_df, test_size=(test_size/(test_size+val_size)), stratify=tmp_df['label'], random_state=seed)
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(out_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    print('Wrote splits to', out_dir, '| counts:', { 'train': len(train_df), 'val': len(val_df), 'test': len(test_df) })

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/raw/alaska2')
    ap.add_argument('--out', default='data/splits')
    a = ap.parse_args()
    main(root=a.root, out_dir=a.out)
