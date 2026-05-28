import os
import json
import glob
import numpy as np
from data_path import DATA_PATH

# loads samples individually to compute mean and std over the dataset using
# rolling averages, in case the dataset exceeds available memory.

data_path=f"{DATA_PATH}/training"
#save_name = os.path.join(data_path, 'dataset_stats.json')
save_name = 'dataset_stats.json'
experimental = True # True -> experimental data, False -> simulated data

files = glob.glob(data_path + "/*.npz")
files.sort()

# ssr = sum of squared residuals divided by (nsamples-1) for variance computation
# signal = features_das = experimental image of reconstructed PA signal
stats = {
    'fluence' : {'min' : np.inf, 'max' : -np.inf, 'mean' : 0.0, 'std' : 0.0, 'ssr' : 0.0},
    'segmentation' : {'min' : np.inf, 'max' : -np.inf, 'mean' : 0.0, 'std' : 0.0, 'ssr' : 0.0},
    'mua' : {'min' : np.inf, 'max' : -np.inf, 'mean' : 0.0, 'std' : 0.0, 'ssr' : 0.0},
    'musp' : {'min' : np.inf, 'max' : -np.inf, 'mean' : 0.0, 'std' : 0.0, 'ssr' : 0.0},
    'signal' : {'min' : np.inf, 'max' : -np.inf, 'mean' : 0.0, 'std' : 0.0, 'ssr' : 0.0}
}
print(f"Found {len(files)} items. computing min, max and means...")
for file_idx, file in enumerate(files):
    print("\r", file_idx+1, "/", len(files), end='', flush=True)
    np_data = np.load(file)
    data = [np_data["fluence"],
            np_data["segmentation"],
            np_data["mua"],
            np_data["musp"]]
    if experimental:
        data.append(np_data["features_das"])
    else:
        data.append(np_data["features_sim"])
    
    for i, stat in enumerate(stats.keys()):
        stats[stat]['min'] = min(stats[stat]['min'], float(np.min(data[i])))
        stats[stat]['max'] = max(stats[stat]['max'], float(np.max(data[i])))
        stats[stat]['mean'] += float(np.mean(data[i]) / len(files))

print("\nComputing std...")
for file_idx, file in enumerate(files):
    print("\r", file_idx+1, "/", len(files), end='', flush=True)
    np_data = np.load(file)
    data = [np_data["fluence"],
            np_data["segmentation"],
            np_data["mua"],
            np_data["musp"]]
    
    if experimental:
        data.append(np_data["features_das"])
    else:
        data.append(np_data["features_sim"])
    
    for i, stat in enumerate(stats.keys()):
        denominator = (np.prod(data[i].shape) * len(files)) - 1
        stats[stat]['ssr'] += np.sum((data[i] - stats[stat]['mean'])**2) / denominator

for stat in stats.keys():
    stats[stat]['std'] = float(np.sqrt(stats[stat]['ssr']))
    
stats['num_samples'] = len(files)
stats['experimental'] = experimental
if stats['segmentation']['min'] == -1:
    stats['segmentation']['min'] += 1
    stats['segmentation']['max'] += 1
    stats['segmentation']['mean'] += 1
    stats['segmentation']['plus_one'] = True
else:
    stats['segmentation']['plus_one'] = False
stats['num_classes'] = int(np.round(stats['segmentation']['max']) + 1)
stats['dx'] = (32e-3) / 300 # 32mm / 300 pixels = 0.00010666666666667 m/pixel
print('\n', json.dumps(stats, indent=4))

# adding the sample naming convention to states as I used for synthetic datasets
stats['normalisation_X'] = [stats['signal']]
stats['normalisation_mu_a'] = [stats['mua']]
stats['normalisation_Phi'] = [stats['fluence']]

with open(save_name, 'w') as f:
    json.dump(stats, f, indent='\t')