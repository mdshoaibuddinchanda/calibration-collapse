"""Fix NSL-KDD: add header, binarize label (normal=0, attack=1)."""
import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, '.')

# NSL-KDD column names (standard)
cols = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

raw = Path('datasets/raw/nsl_kdd.csv')
df  = pd.read_csv(raw, header=None, names=cols)

# Binarize: normal=0, any attack=1
df['label_binary'] = (df['label'] != 'normal').astype(int)
df = df.drop(columns=['label', 'difficulty'])
df = df.rename(columns={'label_binary': 'label'})

# Drop non-numeric categorical columns that need encoding
# (keep numeric features only for simplicity)
cat_cols = ['protocol_type', 'service', 'flag']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Subsample to 50k for memory efficiency (150k is too large for 4GB RAM)
df_sub = df.sample(n=min(50000, len(df)), random_state=42).reset_index(drop=True)

out = Path('datasets/raw/nsl_kdd_processed.csv')
df_sub.to_csv(out, index=False)

vc = df_sub['label'].value_counts()
ir = vc.max() / vc.min()
print(f"NSL-KDD processed: {len(df_sub)} rows, {df_sub.shape[1]-1} features")
print(f"  IR={ir:.1f}, minority_n={vc.min()}, majority_n={vc.max()}")
print(f"  Saved to: {out}")
