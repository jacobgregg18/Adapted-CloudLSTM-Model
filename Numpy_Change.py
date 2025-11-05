import numpy as np
import os
import natsort

def convert_cartesian_to_polar(pc_data_final):
    """
    Convert existing processed data from Cartesian to polar coordinates
    
    Input shape: [N_points, 8]
    - coords: [x,y,z] (normalized by 14500)
    - feats: [cnr, wind_x, wind_y, wind_z] (winds normalized by 10) 
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
    
    # Convert coordinates to polar (unnormalize first, then convert)
    x_actual = x * 14500  # Denormalize
    y_actual = y * 14500
    z_actual = z * 14500
    
    # Calculate polar coordinates (keep unnormalized)
    range_vals = np.sqrt(x_actual**2 + y_actual**2 + z_actual**2)
    azimuth = np.degrees(np.arctan2(y_actual, x_actual))  # In degrees [-180, 180]
    elevation = np.degrees(np.arcsin(z_actual / (range_vals + 1e-8)))  # In degrees [-90, 90]
    
    # Convert wind components to polar (unnormalize first)
    wind_x_actual = wind_x * 10  # Denormalize
    wind_y_actual = wind_y * 10
    wind_z_actual = wind_z * 10
    
    # Calculate radial and tangential wind components
    cos_az = np.cos(np.radians(azimuth))
    sin_az = np.sin(np.radians(azimuth))
    
    wind_radial = wind_x_actual * cos_az + wind_y_actual * sin_az
    wind_tangential = -wind_x_actual * sin_az + wind_y_actual * cos_az
    
    # Keep wind components unnormalized (in original m/s units)
    
    # Combine into new format (all unnormalized, no wind confidence)
    polar_data = np.column_stack([
        azimuth, range_vals, elevation,  # Polar coordinates (unnormalized)
        cnr,  # CNR (already processed)
        wind_radial, wind_tangential, wind_z_actual,  # Polar winds (unnormalized m/s)
        timestamp  # Time (unchanged)
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
        
        # Apply CNR constraint if requested
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
    
    #convert_all_files(input_directory, output_directory, apply_cnr_constraint=False)
    
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
    

    # Load your full file (with azimuth, range, etc.)
    pc = np.load("diff_clouds_polar/2025-07-28_10_40_polar.npy")
    pc2 = np.load("diff_clouds_polar/2025-08-02_10_50_polar.npy")

    print("Shape of data:", pc.shape)

    # Assuming the columns are in the same order as in point_cloud_df:
    # ['sweep_id','time(s)','timestamp','azimuth(deg)','radial_distance(m)',
    #  'elevation(deg)','cnr','X(m)','Y(m)','Z(m)',
    #  'wind_vel_X(m/s)','wind_vel_Y(m/s)','wind_vel_Z(m/s)','wind_vel_confidence(%)']

    az = pc[:, 0]   # azimuth(deg)
    rng = pc[:, 1]  # radial_distance(m)
    elevation = pc[:,2]

    pairs = np.column_stack((az, rng, elevation))

    total = len(pairs)
    unique = len(np.unique(pairs, axis=0))

    print(f"Total pairs: {total}")
    print(f"Unique pairs: {unique}")
    print(f"Duplicates: {total - unique}")
    
    diff = pc2 - pc
    
    print(pc[1000:1010, 0:3])
    print(pc2[1000:1010, 0:3])