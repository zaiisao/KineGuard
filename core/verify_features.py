import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Define paths (using Fragment 3 as our test subject)
matrix_path = '/home/sogang/jaehoon/KineGuard/external/WHAM/output/wham_kineguard/lma_features_id3.npy'
dict_path = '/home/sogang/jaehoon/KineGuard/external/WHAM/output/wham_kineguard/lma_dict_id3.npy'

# 2. Load the data
print("[*] Loading LMA features...")
lma_matrix = np.load(matrix_path)

# Foolproof dictionary extraction
raw_dict_load = np.load(dict_path, allow_pickle=True)

# Unpack the dict whether it's an archive or a 0-D object array
if hasattr(raw_dict_load, 'files'):
    lma_dict = raw_dict_load[raw_dict_load.files[0]].item()
else:
    lma_dict = raw_dict_load.item()

feature_names = sorted(lma_dict.keys())

# Safety check
num_features = lma_matrix.shape[1]
if num_features != 55:
    print(f"[!] Warning: Found {num_features} features instead of 55.")
if len(feature_names) != 55:
    print(f"[!] Warning: Found {len(feature_names)} names instead of 55.")

# 3. Set up a massive figure canvas (11 rows, 5 columns)
fig, axes = plt.subplots(11, 5, figsize=(25, 35))
axes = axes.flatten() 

print("[*] Drawing 55 subplots...")

# 4. Loop through and plot every single feature
for i in range(num_features):
    ax = axes[i]
    feature_data = lma_matrix[:, i]
    
    # Plot the data
    ax.plot(feature_data, color='#1f77b4', linewidth=1.5)
    
    # Labeling and formatting
    ax.set_title(f"[{i:02d}] {feature_names[i]}", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(feature_data))
    
    # Hide X-axis labels unless it's the bottom row to keep it clean
    if i < 50: 
        ax.set_xticklabels([])

# 5. Save the figure to a file
# 5. Save the figure to a file
output_image = '/home/sogang/jaehoon/KineGuard/external/WHAM/output/wham_kineguard/all_55_features_grid.png'

# --- ADD THIS LINE BEFORE SAVING ---
os.makedirs(os.path.dirname(output_image), exist_ok=True)
# -----------------------------------

plt.tight_layout()
plt.savefig(output_image, dpi=150, bbox_inches='tight')
plt.close()

print(f"[SUCCESS] Grid saved to: {os.path.abspath(output_image)}")