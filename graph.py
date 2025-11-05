import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import math
import os
from datetime import datetime

RECON_CNR_VALUE = 0.4
CNR_VALUE = -25

mean_x, std_x = 2.7022033600688215e-06, 0.291664816512239
mean_y, std_y = 4.1019552380086045e-06, 0.2916651588805414
mean_z, std_z = 0.3304346729575734, 0.24687948625791928
cnr_shift = 101.0
cnr_log_mean, cnr_log_std = 4.2543, 0.0588

def unstandardize(tensor, mean, std):
    """Reverses Z-score standardization."""
    return (tensor * std) + mean

def undo_log_normalize_cnr(normalized_cnr):
    """Reverses the log transformation and normalization of CNR."""
    cnr_log = (normalized_cnr * cnr_log_std) + cnr_log_mean
    cnr_shifted = np.exp(cnr_log)
    return cnr_shifted - cnr_shift

def filter_reconstructed(data, radius_threshold_meters):
    """Filter scans reconstructed using xyz coords"""
    
    distance_squared = np.sum(data[:, :3]**2, axis=1)
    combined_mask = (distance_squared < radius_threshold_meters**2) 
    filtered_data = data[combined_mask]
    
    print(np.shape(filtered_data))
        
    x = filtered_data[:, 0]
    x = unstandardize(x, mean_x, std_x)
    y = filtered_data[:, 1]
    y = unstandardize(y, mean_y, std_y)
    z = filtered_data[:, 2]
    z = unstandardize(z, mean_z, std_z)
    cnr = filtered_data[:,3]
    cnr = undo_log_normalize_cnr(cnr)
    mask = cnr > CNR_VALUE

    # Apply the mask to all the arrays to get the filtered data
    filtered_x = x[mask]
    filtered_y = y[mask]
    filtered_z = z[mask]
    
    print(np.shape(filtered_x))
    print(f"max cnr: {np.max(cnr)}, min cnr: {np.min(cnr)}, avg cnr: {np.mean(cnr)}")
    
    return filtered_x, filtered_y, filtered_z

def filter_base(data):
    """Filter base scans with no modifications"""
    
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    cnr = data[:,3]
    mask = cnr > CNR_VALUE

    # Apply the mask to all the arrays to get the filtered data
    filtered_x = x[mask]
    filtered_y = y[mask]
    filtered_z = z[mask]
    filtered_cnr = cnr[mask]
    
    print(np.shape(filtered_x))
    print(f"max cnr: {np.max(cnr)}, min cnr: {np.min(cnr)}, avg cnr: {np.mean(cnr)}")
    
    return filtered_x, filtered_y, filtered_z, filtered_cnr

def filter_rae(data, radius_threshold_meters):
    """Filter scans reconstructed using polar coords"""
    
    mask = data[:, 1] < radius_threshold_meters
    data = data[mask]
    
    azimuth = data[:, 0] * 180
    range_m = data[:, 1]
    elevation = data[:, 2] * 90
    cnr = data[:, 3]

    # Convert to radians
    az_rad = np.deg2rad(azimuth)
    el_rad = np.deg2rad(elevation)

    # Polar to Cartesian
    x = range_m * np.cos(el_rad) * np.cos(az_rad)
    y = range_m * np.cos(el_rad) * np.sin(az_rad)
    z = range_m * np.sin(el_rad)

    print(np.shape(data))
    mask = cnr > RECON_CNR_VALUE

    # Apply the mask to all the arrays to get the filtered data
    filtered_x = x[mask]
    filtered_y = y[mask]
    filtered_z = z[mask]
    filtered_cnr = cnr[mask]
    
    print(np.shape(filtered_x))
    print(f"max cnr: {np.max(filtered_x)}, min cnr: {np.min(filtered_x)}, avg cnr: {np.mean(filtered_x)}")
    print(f"max cnr: {np.max(filtered_y)}, min cnr: {np.min(filtered_y)}, avg cnr: {np.mean(filtered_y)}")
    print(f"max cnr: {np.max(filtered_z)}, min cnr: {np.min(filtered_z)}, avg cnr: {np.mean(filtered_z)}")
    
    return filtered_x, filtered_y, filtered_z, filtered_cnr

#prediction = np.load("LSTM_Scans/LSTM6_prediction103.npy")
prediction = np.load("diff_clouds/2025-08-13_6_10.npy")
#recon_ground = np.load("LSTM_Scans/LSTM_ground3.npy")
recon_ground = np.load("diff_clouds/2025-08-14_2_30.npy")
#previous = np.load("LSTM_Scans/LSTM_prior3.npy")
previous = np.load("diff_clouds/2025-08-14_8_30.npy")


fig = plt.figure(figsize=(15, 5))
x_limits = (-0.6, 0.6)
y_limits = (-0.6,0.6)
z_limits = (0,0.2)

# Prediction plot
#x, y, z, cnr = filter_rae(prediction, 6)
x, y, z, cnr = filter_base(prediction)

# Define thresholds
bins = [0.4, 0.5, 0.6, 1.1]
bins = [-25, -15, -10, 10]
labels = ['CNR [-20,-15]', 'CNR [-15,-10]', 'CNR [-10,10]']
colors = ['blue', 'orange', 'red']

# Assign color by bin
cnr_bins = np.digitize(cnr, bins) - 1
point_colors = [colors[i] for i in cnr_bins]

ax1 = fig.add_subplot(133, projection='3d')
for label, color in zip(labels, colors):
    mask = np.array(point_colors) == color
    ax1.scatter(x[mask], y[mask], z[mask], c=color, s=1, label=label)

ax1.set_title('Predicted Scan')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_xlim(x_limits)
ax1.set_ylim(y_limits)
ax1.set_zlim(z_limits)

# Ground Truth plot
# x,y,z,cnr = filter_rae(recon_ground, 6)
x, y, z, cnr = filter_base(recon_ground)
ax1 = fig.add_subplot(132, projection='3d')  # 1 row, 3 columns, first plot

# Assign color by bin
cnr_bins = np.digitize(cnr, bins) - 1
point_colors = [colors[i] for i in cnr_bins]
for label, color in zip(labels, colors):
    mask = np.array(point_colors) == color
    ax1.scatter(x[mask], y[mask], z[mask], c=color, s=1, label=label)
    
ax1.set_title('Target Scan')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_xlim(x_limits)
ax1.set_ylim(y_limits)
ax1.set_zlim(z_limits)


# Previous Plot
#x,y,z,cnr = filter_rae(previous, 6)
x, y, z, cnr = filter_base(previous)
ax1 = fig.add_subplot(131, projection='3d')  # 1 row, 3 columns, first plot

# Assign color by bin
cnr_bins = np.digitize(cnr, bins) - 1
point_colors = [colors[i] for i in cnr_bins]
for label, color in zip(labels, colors):
    mask = np.array(point_colors) == color
    ax1.scatter(x[mask], y[mask], z[mask], c=color, s=1, label=label)
ax1.set_title('Previous Scan')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_xlim(x_limits)
ax1.set_ylim(y_limits)
ax1.set_zlim(z_limits)

fig.legend(markerscale=5)
plt.show()