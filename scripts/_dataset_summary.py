import sys, os
sys.path.insert(0, '.')
import pandas as pd

print("=== REAL DATASETS ===")
real = [
    ('pima',        'datasets/raw/pima.csv',        'Outcome'),
    ('phoneme',     'datasets/raw/phoneme.csv',      'Class'),
    ('credit_card', 'datasets/raw/credit_card.csv',  'Class'),
]
for name, path, target in real:
    df = pd.read_csv(path)
    vc = df[target].value_counts()
    ir = vc.max() / vc.min()
    print(f"  {name}: rows={len(df)}, features={df.shape[1]-1}, IR={ir:.1f}, minority_n={vc.min()}, majority_n={vc.max()}")

print()
print("=== SYNTHETIC DATASETS ===")
syn_dir = 'datasets/synthetic'
for f in sorted(os.listdir(syn_dir)):
    if not f.endswith('.csv'):
        continue
    df = pd.read_csv(f'{syn_dir}/{f}')
    vc = df['label'].value_counts()
    ir = vc.max() / vc.min()
    print(f"  {f}: rows={len(df)}, features={df.shape[1]-1}, IR={ir:.1f}, minority_n={vc.min()}")
