import sys, pandas as pd
sys.path.insert(0, '.')
datasets = [
    ('pima',                        'datasets/raw/pima.csv',                                              'Outcome'),
    ('phoneme',                     'datasets/raw/phoneme.csv',                                           'Class'),
    ('credit_card',                 'datasets/raw/credit_card.csv',                                       'Class'),
    ('confidence_collapse_severe',  'datasets/synthetic/confidence_collapse_severe_n5000_seed42.csv',     'label'),
    ('extreme_imbalance_severe',    'datasets/synthetic/extreme_imbalance_severe_n5000_seed42.csv',       'label'),
    ('boundary_overlap_severe',     'datasets/synthetic/boundary_overlap_severe_n5000_seed42.csv',        'label'),
]
for name, path, target in datasets:
    df = pd.read_csv(path)
    vc = df[target].value_counts()
    ir = vc.max()/vc.min()
    print(f"{name}: rows={len(df)}, features={df.shape[1]-1}, IR={ir:.1f}, minority_n={vc.min()}")
