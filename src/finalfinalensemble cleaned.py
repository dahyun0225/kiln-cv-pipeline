print(f"{'>'*70}")
"""
Final Final Ensembles 6 variations from top performing models
"""
import pandas as pd
import numpy as np


csv_files = {
    '5cvs': 'submission411_5cv_s.csv',              # 5-fold CV (99.944%)
    '5cvf': 'submission411_5cv_f.csv',              # 5-fold CV + full (99.944%)
    'eff1': 'submission4_efficientnet1 99686.csv',  # highest single model (99.686%)
    'eff0': 'submission4_efficientnet0 9958.csv',   # EfficientNet0 (99.58% for variation)
}

print("\nread CSVs")
dfs = {}
for name, filepath in csv_files.items():
    try:
        df = pd.read_csv(filepath)
        print(f"  {name}: {len(df)} rows, score range [{df['score'].min():.4f}, {df['score'].max():.4f}]")
        dfs[name] = df
    except FileNotFoundError:
        print(f"  {name}: FILE NOT FOUND - {filepath}")

if len(dfs) == 0:
    print("\nERROR: No CSV files found")
    exit(1)

# check row indx match
base_index = list(dfs.values())[0]['index'].values
for name, df in dfs.items():
    if not np.array_equal(df['index'].values, base_index):
        print(f"Check: {name} has different indices!")
        raise ValueError(f"Index mismatch in {name}")



# V1: 3 best performers (5cvs + 5cvf + eff1) -> I got 99.951% with this one
print(f"\n{'-'*60}")
print("V1: Top 3 (5cvs + 5cvf + eff1)")
var1_scores = np.zeros(len(base_index))
var1_models = ['5cvs', '5cvf', 'eff1']
models_found = []
for model in var1_models:
    if model in dfs:
        var1_scores += dfs[model]['score'].values
        models_found.append(model)

if len(models_found) > 0:
    var1_scores /= len(models_found)
    var1_df = pd.DataFrame({'index': base_index, 'score': var1_scores})
    var1_df.to_csv("finalensemble_var1_top3.csv", index=False)
    print(f"  Saved: finalensemble_var1_top3.csv")
    print(f"  Models: {', '.join(models_found)}")
    print(f"  Mean: {var1_scores.mean():.4f}, Std: {var1_scores.std():.4f}")

# V2: all 4 models (5cvs + 5cvf + eff1 + eff0)
print(f"\n{'-'*60}")
print("V2: 4 models (5cvs + 5cvf + eff1 + eff0)")
var2_scores = np.zeros(len(base_index))
var2_models = ['5cvs', '5cvf', 'eff1', 'eff0']
models_found = []
for model in var2_models:
    if model in dfs:
        var2_scores += dfs[model]['score'].values
        models_found.append(model)

if len(models_found) > 0:
    var2_scores /= len(models_found)
    var2_df = pd.DataFrame({'index': base_index, 'score': var2_scores})
    var2_df.to_csv("finalensemble_var2_all4.csv", index=False)
    print(f"  Saved: finalensemble_var2_all4.csv")
    print(f"  Models: {', '.join(models_found)}")
    print(f"  Mean: {var2_scores.mean():.4f}, Std: {var2_scores.std():.4f}")

# V3: both EfficientNets (eff1 + eff0)
print(f"\n{'-'*60}")
print("V3: 2 EfficientNets (eff1 + eff0)")
var3_scores = np.zeros(len(base_index))
var3_models = ['eff1', 'eff0']
models_found = []
for model in var3_models:
    if model in dfs:
        var3_scores += dfs[model]['score'].values
        models_found.append(model)

if len(models_found) > 0:
    var3_scores /= len(models_found)
    var3_df = pd.DataFrame({'index': base_index, 'score': var3_scores})
    var3_df.to_csv("finalensemble_var3_effnets.csv", index=False)
    print(f"  Saved: finalensemble_var3_effnets.csv")
    print(f"  Models: {', '.join(models_found)}")
    print(f"  Mean: {var3_scores.mean():.4f}, Std: {var3_scores.std():.4f}")

# V4: 5-fold CVs + eff0 (5cvs + 5cvf + eff0)
print(f"\n{'-'*60}")
print("V4: 5-fold CVs + eff0 (5cvs + 5cvf + eff0)")
var4_scores = np.zeros(len(base_index))
var4_models = ['5cvs', '5cvf', 'eff0']
models_found = []
for model in var4_models:
    if model in dfs:
        var4_scores += dfs[model]['score'].values
        models_found.append(model)

if len(models_found) > 0:
    var4_scores /= len(models_found)
    var4_df = pd.DataFrame({'index': base_index, 'score': var4_scores})
    var4_df.to_csv("finalensemble_var4_5cv_eff0.csv", index=False)
    print(f"  Saved: finalensemble_var4_5cv_eff0.csv")
    print(f"  Models: {', '.join(models_found)}")
    print(f"  Mean: {var4_scores.mean():.4f}, Std: {var4_scores.std():.4f}")

# V5: Weighted ensemble
print("V5: w top 3 (40% eff1, 30% 5cvf, 30% 5cvs)")
var5_scores = np.zeros(len(base_index))
weights = {'eff1': 0.4, '5cvf': 0.3, '5cvs': 0.3}
models_found = []
for model, weight in weights.items():
    if model in dfs:
        var5_scores += dfs[model]['score'].values * weight
        models_found.append(f"{model}({weight})")

if len(models_found) > 0:
    var5_df = pd.DataFrame({'index': base_index, 'score': var5_scores})
    var5_df.to_csv("finalensemble_var5_weighted.csv", index=False)
    print(f"  Saved: finalensemble_var5_weighted.csv")
    print(f"  Weights: {', '.join(models_found)}")
    print(f"  Mean: {var5_scores.mean():.4f}, Std: {var5_scores.std():.4f}")

# V6: different weighted ensemble but CV emphasized
print("V6: CV-weighted (35% 5cvs, 35% 5cvf, 30% eff1)")
var6_scores = np.zeros(len(base_index))
weights2 = {'5cvs': 0.35, '5cvf': 0.35, 'eff1': 0.30}
models_found = []
for model, weight in weights2.items():
    if model in dfs:
        var6_scores += dfs[model]['score'].values * weight
        models_found.append(f"{model}({weight})")

if len(models_found) > 0:
    var6_df = pd.DataFrame({'index': base_index, 'score': var6_scores})
    var6_df.to_csv("finalensemble_var6_cv_weighted.csv", index=False)
    print(f"  Saved: finalensemble_var6_cv_weighted.csv")
    print(f"  Weights: {', '.join(models_found)}")
    print(f"  Mean: {var6_scores.mean():.4f}, Std: {var6_scores.std():.4f}")
