import numpy as np
import os
import natsort

def convert_cartesian_to_polar(pc_data_final):
    """
    Convert existing processed data from Cartesian to polar coordinates
    
    Input shape: [N_points, 8]
    - coords: [x,y,z]
    - feats: [cnr, wind_x, wind_y, wind_z]
    - time: [timestamp]
    
    Output shape: [N_points, 7] 
    - polar_coords: [azimuth, range, elevation]
    - polar_feats: [cnr, wind_radial, wind_tangential, wind_z]
    - time: [timestamp]
    """
    
    # Extract components
    x, y, z = pc_data_final[:, 0], pc_data_final[:, 1], pc_data_final[:, 2]
    cnr = pc_data_final[:, 3]
    wind_x, wind_y, wind_z = pc_data_final[:, 4], pc_data_final[:, 5], pc_data_final[:, 6]
    timestamp = pc_data_final[:, 7]
    
    # Convert coordinates to polar
    x_actual = x * 14500
    y_actual = y * 14500
    z_actual = z * 14500
    
    # Calculate polar coordinates
    range_vals = np.sqrt(x_actual**2 + y_actual**2 + z_actual**2)
    azimuth = np.degrees(np.arctan2(y_actual, x_actual))  # In degrees [-180, 180]
    elevation = np.degrees(np.arcsin(z_actual / (range_vals + 1e-8)))  # In degrees [-90, 90]
    
    # Convert wind components to polar
    wind_x_actual = wind_x * 10
    wind_y_actual = wind_y * 10
    wind_z_actual = wind_z * 10
    
    # Calculate radial and tangential wind components
    cos_az = np.cos(np.radians(azimuth))
    sin_az = np.sin(np.radians(azimuth))
    
    wind_radial = wind_x_actual * cos_az + wind_y_actual * sin_az
    wind_tangential = -wind_x_actual * sin_az + wind_y_actual * cos_az
    
    # Combine into new format (all unnormalized, no wind confidence)
    polar_data = np.column_stack([
        azimuth, range_vals, elevation,
        cnr,
        wind_radial, wind_tangential, wind_z_actual,
        timestamp
    ])
    
    return polar_data

def apply_cnr_range_constraint(polar_data, cnr_min=-50, cnr_max=20):
    """
    Apply your supervisor's [-20, 20] CNR range constraint
    """
    cnr = polar_data[:, 3]  # CNR is 4th column
    
    # Clip CNR values
    cnr_clipped = np.clip(cnr, cnr_min, cnr_max)
    
    # Normalize to [0,1] range  
    cnr_normalized = (cnr_clipped - cnr_min) / (cnr_max - cnr_min)
    
    # Replace CNR column
    polar_data_constrained = polar_data.copy()
    polar_data_constrained[:, 3] = cnr_normalized
    
    return polar_data_constrained

def convert_all_files(input_dir, output_dir, apply_cnr_constraint=True):
    """
    Convert all existing .npy files to polar coordinates
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = natsort.natsorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    
    for filename in files:
        print(f"Processing {filename}...")
        
        # Load existing data
        pc_data = np.load(os.path.join(input_dir, filename))
        
        # Convert to polar
        polar_data = convert_cartesian_to_polar(pc_data)
        
        # Apply CNR constraint if needed
        if apply_cnr_constraint:
            polar_data = apply_cnr_range_constraint(polar_data)
        
        # Save with new name
        output_filename = filename.replace('.npy', '_polar.npy')
        np.save(os.path.join(output_dir, output_filename), polar_data)
        
        print(f"  Converted shape: {pc_data.shape} -> {polar_data.shape}")
        print(f"  CNR range: [{polar_data[:, 3].min():.3f}, {polar_data[:, 3].max():.3f}]")

# Usage example
if __name__ == "__main__":
    lidar_dir = os.getcwd()
    input_directory = os.path.join(lidar_dir, "diff_clouds")
    output_directory = os.path.join(lidar_dir, "diff_clouds_polar")
    
    convert_all_files(input_directory, output_directory, apply_cnr_constraint=False)
    
    print("Conversion complete! New data format (all unnormalized, no wind confidence):")
    print("- Column 0: Azimuth (degrees, -180 to 180)")
    print("- Column 1: Range (meters, 0 to 14500)")  
    print("- Column 2: Elevation (degrees, -90 to 90)")
    print("- Column 3: CNR (original units, with optional [-20,20] constraint)")
    print("- Column 4: Wind Radial (m/s)")
    print("- Column 5: Wind Tangential (m/s)")
    print("- Column 6: Wind Vertical (m/s)")
    print("- Column 7: Timestamp")
    print("\nYou can normalize these later in your training code as needed!")