import matplotlib.pyplot as plt
import numpy as np

# Data points across domains
medical_x = [0.427, 0.320, 0.417, 0.317, 0.323, 0.391, 0.365, 0.328]
medical_y = [1.147, 0.598, 0.605, 0.346, 0.969, 0.371, 0.316, 0.302]

philosophy_x = [0.331, 0.338, 0.269, 0.283, 0.302, 0.266]
philosophy_y = [0.527, 0.638, 0.314, 0.357, 0.867, 0.314]

legal_x = [0.276, 0.209, 0.265, 0.252, 0.206]
legal_y = [1.147, 1.263, 1.568, 1.191, 1.428]

ethics_x = [0.257, 0.181, 0.211, 0.183, 0.172]
ethics_y = [0.920, 0.925, 0.794, 1.027, 1.191]

# K Constants per domain
k_med = 0.429
k_leg = 0.348
k_phi = 0.301
k_eth = 0.193

# Setup figure parameters
plt.figure(figsize=(10, 8), dpi=300, facecolor='white')

# Generate scatter plots
plt.scatter(medical_x, medical_y, color='red', label='Medical', s=70, alpha=0.8, edgecolor='k')
plt.scatter(philosophy_x, philosophy_y, color='blue', label='Philosophy', s=70, alpha=0.8, edgecolor='k')
plt.scatter(legal_x, legal_y, color='green', label='Legal', s=70, alpha=0.8, edgecolor='k')
plt.scatter(ethics_x, ethics_y, color='orange', label='Ethics', s=70, alpha=0.8, edgecolor='k')

# Generate Hyperbolas representing the Conservation Constraint
x_vals = np.linspace(0.15, 0.45, 200)

y_med = k_med / x_vals
y_phi = k_phi / x_vals
y_leg = k_leg / x_vals
y_eth = k_eth / x_vals

plt.plot(x_vals, y_med, 'r--', alpha=0.7)
plt.plot(x_vals, y_phi, 'b--', alpha=0.7)
plt.plot(x_vals, y_leg, 'g--', alpha=0.7)
plt.plot(x_vals, y_eth, '--', color='orange', alpha=0.7)

# Overlay Annotations along curves
plt.text(0.41, k_med/0.41 + 0.05, f'K = {k_med}', color='red', fontsize=11, ha='center', weight='bold')
plt.text(0.39, k_phi/0.39 + 0.05, f'K = {k_phi}', color='blue', fontsize=11, ha='center', weight='bold')
plt.text(0.35, k_leg/0.35 + 0.05, f'K = {k_leg}', color='green', fontsize=11, ha='center', weight='bold')
plt.text(0.25, k_eth/0.25 + 0.05, f'K = {k_eth}', color='orange', fontsize=11, ha='center', weight='bold')

# Axial limits
plt.xlim(0.15, 0.45)
plt.ylim(0.2, 2.0)

# Labels
plt.xlabel('ΔRCI (Delta Relational Coherence Index)', fontsize=12, fontweight='bold')
plt.ylabel('Var_Ratio (Variance Ratio)', fontsize=12, fontweight='bold')

# Titling
plt.suptitle('Conservation Constraint: ΔRCI × Var_Ratio ≈ K(domain)', fontsize=16, y=0.96, fontweight='bold')
plt.title('24 model-domain runs across 14 architectures from 8 vendors', fontsize=12, pad=10)

plt.legend(fontsize=11, loc='upper right', framealpha=0.9)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Export figure
plt.savefig('figure1_conservation_constraint.png', dpi=300, bbox_inches='tight')
print("Saved: figure1_conservation_constraint.png")
