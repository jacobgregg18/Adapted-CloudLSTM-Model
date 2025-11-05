import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial import cKDTree

# --- DATA LOADING AND PROCESSING ---
# --- DATA LOADING AND PROCESSING ---
def load_and_process_predictions(pred_file_path, gt_file_path, radius_threshold_meters):
    """
    Loads predicted and ground truth features from numpy files, aligns them
    using a nearest-neighbor search, and creates a dictionary for plotting.

    Args:
        pred_file_path (str): The path to the .npy file containing predictions.
        gt_file_path (str): The path to the .npy file containing ground truth.

    Returns:
        dict: A dictionary containing 'groundTruth' and 'predicted' data,
              or None if a file is not found.
    """
    # Check if both files exist
    if not os.path.exists(pred_file_path):
        print(f"Error: The prediction file '{pred_file_path}' was not found.")
        print("Please ensure the validation script has been run to create this file.")
        return None
    
    if not os.path.exists(gt_file_path):
        print(f"Error: The ground truth file '{gt_file_path}' was not found.")
        return None

    try:
        # Load both the predicted and ground truth numpy arrays.
        # The .squeeze(axis=0) will remove the first dimension if it is of size 1.
        pred_data = np.load(pred_file_path)
        gt_data = np.load(gt_file_path)

        print(f"Successfully loaded prediction data with shape {pred_data.shape}.")
        print(f"Successfully loaded ground truth data with shape {gt_data.shape}.")

        # Check if the loaded data has the expected number of features (8).
        if pred_data.shape[-1] != 8 or gt_data.shape[-1] != 8:
            print(f"Error: Expected 8 features in both files.")
            #return None

        # Check if the number of points matches
        if pred_data.shape[0] != gt_data.shape[0]:
            print("Error: The number of points in the prediction and ground truth files do not match.")
            #return None

    except Exception as e:
        print(f"An error occurred while loading the files: {e}")
        return None

    # --- Crucial step: Align the predicted data with the ground truth data ---

    # Unstandardize XYZ and time using their respective means and stds
    #gt_data[:,0] = unstandardize(gt_data[:,0], mean_x, std_x)
    #gt_data[:,1] = unstandardize(gt_data[:,1], mean_y, std_y)
    #gt_data[:,2] = unstandardize(gt_data[:,2], mean_z, std_z)
    #gt_data[:,7] = unstandardize(gt_data[:,7], mean_time, std_time)

    # Undo the log normalization for CNR
    #gt_data[:,3] = undo_log_normalize_cnr(gt_data[:,3], cnr_shift, cnr_log_mean, cnr_log_std)

    # Undo the normalization for wind components
    #gt_data[:,4] = undo_normalize_winds(gt_data[:,4], wind_scale)
    #gt_data[:,5] = undo_normalize_winds(gt_data[:,5], wind_scale)
    #gt_data[:,6] = undo_normalize_winds(gt_data[:,6], wind_scale)
    
    # Unstandardize XYZ and time using their respective means and stds
    #pred_data[:,0] = unstandardize(pred_data[:,0], mean_x, std_x)
    #pred_data[:,1] = unstandardize(pred_data[:,1], mean_y, std_y)
    #pred_data[:,2] = unstandardize(pred_data[:,2], mean_z, std_z)
    #pred_data[:,7] = unstandardize(pred_data[:,7], mean_time, std_time)

    # Undo the log normalization for CNR
    #pred_data[:,3] = undo_log_normalize_cnr(pred_data[:,3], cnr_shift, cnr_log_mean, cnr_log_std)

    # Undo the normalization for wind components
    #pred_data[:,4] = undo_normalize_winds(pred_data[:,4], wind_scale)
    #pred_data[:,5] = undo_normalize_winds(pred_data[:,5], wind_scale)
    #pred_data[:,6] = undo_normalize_winds(pred_data[:,6], wind_scale)

    #distance_squared = np.sum(gt_data[:, :3]**2, axis=1)

    # Create a combined boolean mask that includes:
    # 1. Points with a squared distance less than the squared radius.
    # 2. Points where the wind_x value is within a specific range.
    #combined_mask = (distance_squared < radius_threshold_meters**2)
    
    #radial_distance = np.sqrt(pred_data[:, 0]**2 + pred_data[:, 1]**2 + pred_data[:, 2]**2)
    #pred_data = pred_data[radial_distance <= radius_threshold_meters]
    #filtered_data = gt_data[combined_mask]
    
    filtered_data = gt_data[gt_data[:,1] < radius_threshold_meters]
    pred_data = pred_data[gt_data[:,1] < radius_threshold_meters]

    print(f"New shape of predicted {pred_data.shape} and new shape of ground {filtered_data.shape}")
    
    filtered_data[:,3] = (filtered_data[:,3] * 50) - 40
    pred_data[:,3] = (pred_data[:,3] * 50) - 40
    # Define the features and their corresponding indices in the numpy array
    # The loaded data has the following columns:
    # 0: x, 1: y, 2: z, 3: CNR, 4: wind_x, 5: wind_y, 6: wind_z, 7: time
    
    # Slice the predicted features
    predicted_features = {
        'x': pred_data[..., 0],
        'y': pred_data[..., 1],
        'z': pred_data[..., 2],
        'CNR': pred_data[..., 3],
        'wind_x': pred_data[..., 4],
        'wind_y': pred_data[..., 5],
        'wind_z': pred_data[..., 6]
        #'time': pred_data[..., 7]
    }

    # Slice the ground truth features using the aligned data
    ground_truth_features = {
        'x': filtered_data[..., 0],
        'y': filtered_data[..., 1],
        'z': filtered_data[..., 2],
        'CNR': filtered_data[..., 3],
        'wind_x': filtered_data[..., 4],
        'wind_y': filtered_data[..., 5],
        'wind_z': filtered_data[..., 6]
        #'time': filtered_data[..., 7]
    }
    
    return {
        'groundTruth': ground_truth_features,
        'predicted': predicted_features
    }

# --- DE-STANDARDIZATION LOGIC ---

# Placeholder values for standardization statistics.
# YOU MUST REPLACE THESE WITH THE ACTUAL VALUES FROM YOUR TRAINING DATA.
# For example: mean_cnr = -15.5
mean_x, std_x = 2.7022033600688215e-06, 0.291664816512239
mean_y, std_y = 4.1019552380086045e-06, 0.2916651588805414
mean_z, std_z = 0.3304346729575734, 0.24687948625791928
mean_time, std_time = 1754092671.0763593, 220765.19726839956
cnr_shift = 101.0
cnr_log_mean, cnr_log_std = 4.2543, 0.0588
wind_scale = 0.15

def unstandardize(tensor, mean, std):
    """Reverses Z-score standardization."""
    return (tensor * std) + mean

def undo_log_normalize_cnr(normalized_cnr, cnr_shift, cnr_log_mean, cnr_log_std):
    """Reverses the log transformation and normalization of CNR."""
    cnr_log = (normalized_cnr * cnr_log_std) + cnr_log_mean
    cnr_shifted = np.exp(cnr_log)
    return cnr_shifted - cnr_shift

def undo_normalize_winds(normalized_winds, scale):
    """
    Reverses the normalization of wind components and sets values close to zero to zero.
    """
    # First, perform the un-normalization
    unnormalized_winds = normalized_winds * scale
    
    # Create a boolean mask for values below 1e-5
    # Use np.abs to handle both positive and negative values close to zero
    zero_mask = np.abs(unnormalized_winds) < 1e-5
    
    # Set the values at the masked positions to 0
    unnormalized_winds[zero_mask] = 0.0
    
    return unnormalized_winds

def unstandardize_predicted_features(predicted_features_dict):
    """
    Applies de-standardization to the predicted features.
    
    Args:
        predicted_features_dict (dict): The dictionary of predicted features.
        
    Returns:
        dict: A new dictionary with unstandardized features.
    """
    unstandardized_dict = predicted_features_dict.copy()

    # Unstandardize XYZ and time using their respective means and stds
    unstandardized_dict['x'] = unstandardized_dict['x']
    unstandardized_dict['y'] = unstandardized_dict['y']
    unstandardized_dict['z'] = unstandardized_dict['z']
    #unstandardized_dict['time'] = unstandardized_dict['time']

    # Undo the log normalization for CNR
    unstandardized_dict['CNR'] = unstandardized_dict['CNR']

    # Undo the normalization for wind components
    unstandardized_dict['wind_x'] = unstandardized_dict['wind_x']
    unstandardized_dict['wind_y'] = unstandardized_dict['wind_y']
    unstandardized_dict['wind_z'] = unstandardized_dict['wind_z']
    
    return unstandardized_dict

# --- PLOTTING LOGIC ---
def plot_performance(data, selected_features):
    """
    Creates and displays charts for the selected features' performance.

    Args:
        data (dict): The dataset containing ground truth and predicted values.
        selected_features (list): A list of features to analyze and plot.
    """
    if data is None:
        print("Cannot plot. Data is not available.")
        return
        
    num_features = len(selected_features)
    # Create a figure with a grid of subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Performance Analysis of All Features", fontsize=20)
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    print(data['predicted']['time'].mean())
    print(data['predicted']['time'].max())
    print(data['predicted']['time'].min())
    print(data['groundTruth']['time'].mean())
    print(data['groundTruth']['time'].max())
    print(data['groundTruth']['time'].min())

    for i, feature in enumerate(selected_features):
        ax = axes[i]
        
        if feature not in data['predicted']:
            print(f"Error: Feature '{feature}' not found in data. Skipping.")
            continue

        # Plot Overlaid Histograms for Ground Truth and Predicted Data
        ax.hist(data['groundTruth'][feature], bins=50, alpha=0.7, color='#3b82f6', label='Ground Truth')
        ax.hist(data['predicted'][feature], bins=50, alpha=0.7, color='#86efac', label='Predicted')
        
        ax.set_title(f"Distribution for '{feature}'")
        ax.set_xlabel(f"{feature} Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_cnr_subplots(gt_data, pred_datasets, labels):
    """
    Creates and displays a grid of histograms for CNR, with each subplot
    comparing a single predicted dataset against the ground truth.

    Args:
        gt_data (dict): The ground truth dataset.
        pred_datasets (list): A list of dictionaries, where each dict is a
                              predicted dataset.
        labels (list): A list of strings for the legend, corresponding to
                       each predicted dataset.
    """
    if not gt_data or not pred_datasets or not labels:
        print("Cannot plot. Data or labels are missing.")
        return
        
    # Create a 2x2 grid of subplots for 4 histograms
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("CNR Prediction Comparison", fontsize=20)
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    colors = ['#86efac', '#fde047', '#fca5a5', '#94a3b8']
    for i, pred_data in enumerate(pred_datasets):
        ax = axes[i]
        
        # Plot the Ground Truth CNR as a solid baseline
        ax.hist(gt_data['CNR'], bins=50, alpha=0.7, color='#3b82f6', label='Ground Truth')
        
        # Plot the predicted CNR
        ax.hist(pred_data['CNR'], bins=50, alpha=0.7, color=colors[i], label=labels[i])
            
        ax.set_title(f"Distribution for {labels[i]}")
        ax.set_xlabel("CNR Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_multiple_cnr_subplots(gt_datasets, pred_datasets, labels):
    """
    Creates and displays a grid of histograms for CNR, with each subplot
    comparing a single predicted dataset against the ground truth.

    Args:
        gt_datasets (list): A list of dictionaries, where each dict is a
                            ground truth dataset.
        pred_datasets (list): A list of dictionaries, where each dict is a
                              predicted dataset.
        labels (list): A list of strings for the legend, corresponding to
                       each predicted dataset.
    """
    if not gt_datasets or not pred_datasets or not labels:
        print("Cannot plot. Data or labels are missing.")
        return
        
    # Create a 2x2 grid of subplots for 4 histograms
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("CNR Prediction Comparison", fontsize=20)
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    colors = ['#86efac', '#fde047', '#fca5a5', '#e61212', "#0D0729", "#94a3b8", "#d129e0", "#507BF3", "#e68312"]
    for i in range(len(pred_datasets)):
        ax = axes[i]
        
        bins = np.linspace(-40,10,100)
        
        # Plot the Ground Truth CNR as a solid baseline
        ax.hist(gt_datasets[i]['CNR'], bins=bins, alpha=0.7, color='#3b82f6', label='Ground Truth')
        
        # Plot the predicted CNR
        ax.hist(pred_datasets[i]['CNR'], bins=bins, alpha=0.7, color=colors[i], label=labels[i])
            
        ax.set_title(f"Distribution for {labels[i]}")
        ax.set_xlabel("CNR Value (dB)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# --- MAIN SCRIPT EXECUTION ---
if __name__ == "__main__":
    # Specify the paths to your prediction and ground truth files
    gt_file_paths = [
        "Recreated_Scans\Train_offset_ground0.npy",   #  Train 0, test 2
        "Recreated_Scans\Train_offset_ground0.npy",  # Train 1, test 1
        "Recreated_Scans\Train_offset_ground1.npy",  # 
        "Recreated_Scans\Train_offset_ground1.npy",
        "Recreated_Scans\Train_offset_ground2.npy",  # 
        "Recreated_Scans\Train_offset_ground2.npy"   # 
    ]
    
    gt_file_paths = [
        "LSTM_ground2.npy",   #  Train 0, test 2
        "LSTM_ground2.npy",  # Train 1, test 1
        "LSTM_ground2.npy",  # 
        "LSTM_ground2.npy",
        "LSTM_ground2.npy",  # 
        "LSTM_ground2.npy",
        "LSTM_ground2.npy",
        "LSTM_ground2.npy",  # 
        "LSTM_ground2.npy" 
    ]
    
    gt_file_paths = [ 
        "LSTM_ground0.npy",
        "LSTM_ground1.npy",
        "LSTM_ground2.npy",  # 
        "LSTM_ground3.npy" 
    ]
    
    
    # Define the file paths for each of your four predicted datasets
    pred_file_paths = [
        "Test_offset37_prediction3_0.npy",
        "Test_offset39_prediction3_0.npy",
        "Test_offset37_prediction3_1.npy", 
        "Test_offset39_prediction3_1.npy",
        "Test_offset37_prediction3_2.npy", 
        "Test_offset39_prediction3_2.npy" 
    ]
    
    pred_file_paths = [
        "LSTM_Scans/LSTM_prediction12.npy",
        "LSTM_Scans/LSTM_prediction22.npy",
        "LSTM_Scans/LSTM_prediction32.npy",
        "LSTM_Scans/LSTM_prediction42.npy",
        "LSTM_Scans/LSTM_prediction52.npy",
        "LSTM_Scans/LSTM_prediction62.npy",
        "LSTM_Scans/LSTM_prediction72.npy",
        "LSTM_Scans/LSTM_prediction82.npy",
        "LSTM_Scans/LSTM_prediction92.npy"
    ]
    
    pred_file_paths = [
        "LSTM_Scans/LSTM22_prediction100.npy",
        "LSTM_Scans/LSTM22_prediction101.npy",
        "LSTM_Scans/LSTM22_prediction102.npy",
        "LSTM_Scans/LSTM22_prediction103.npy"
    ]
    
    radius_threshold_meters = 1.25
    
    gt_datasets = []
    predicted_datasets = []
    
    # Load, process, and de-standardize all ground truth and prediction datasets
    for i in range(len(pred_file_paths)):
        pred_path = pred_file_paths[i]
        gt_path = gt_file_paths[i]
        
        # Load and process the data for each pair
        data_dict = load_and_process_predictions(pred_path, gt_path, radius_threshold_meters)
        
        if data_dict:
            # De-standardize both the ground truth and predicted data
            gt_data = unstandardize_predicted_features(data_dict['groundTruth'])
            pred_data = unstandardize_predicted_features(data_dict['predicted'])
            
            gt_datasets.append(gt_data)
            predicted_datasets.append(pred_data)

    # Define the labels for each prediction model
    #prediction_labels = ["Time Stamp 1", "Time Stamp 2", "Time Stamp 3", "Time Stamp 4"]
    #prediction_labels = ["Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4", "Epoch 5", "Epoch 6", "Epoch 7", "Epoch 8", "Epoch 9"]
    prediction_labels = ["Epoch 7", "Epoch 8", "Epoch 9", "Predicted Values"]
    #prediction_labels = ["Scan 1", "Scan 2", "Scan 3", "Scan 4", "Scan 3", "Scan 3"]
    
    # Now, plot the comparison of CNR predictions in separate subplots
    plot_multiple_cnr_subplots(gt_datasets, predicted_datasets, prediction_labels)
    
    print("\nPlotted CNR comparison for all requested models.")
"""
# --- MAIN SCRIPT EXECUTION ---
if __name__ == "__main__":
    # Specify the paths to your prediction and ground truth files
    pred_file_path = "Test_offset38_prediction3_3.npy"
    gt_file_path = "Test_offset39_prediction3_3.npy"
    #gt_file_path = "Test_offset6_prediction.npy"
    
    # Load and process the data
    data_for_plotting = load_and_process_predictions(pred_file_path, gt_file_path, 6)
    
    if data_for_plotting:
        data_for_plotting['groundTruth'] = unstandardize_predicted_features(data_for_plotting['groundTruth'])
        data_for_plotting['predicted'] = unstandardize_predicted_features(data_for_plotting['predicted'])

        # Define the list of features you want to plot
        features_to_plot = ['CNR', 'wind_x', 'wind_y', 'wind_z', 'time']
        plot_performance(data_for_plotting, selected_features=features_to_plot)
        
        print("\nPlotted performance for all requested features.")

if __name__ == "__main__":
    gt_file_path = "Recreated_Scans\Train_offset_ground2.npy"
    
    # Define the file paths for each of your four predicted datasets
    pred_file_paths = [
        "Test_offset22_prediction0.npy",
        "Test_offset25_prediction0.npy",  # <-- REPLACE WITH YOUR FILE PATH
        "Test_offset17_prediction0.npy",
        "Test_offset24_prediction.npy"
    ]
    radius_threshold_meters = 6
    
    # Load and process the ground truth data
    # Note: We only need to load the ground truth once
    gt_data_dict = load_and_process_predictions(pred_file_paths[0], gt_file_path, radius_threshold_meters)
    
    if gt_data_dict:
        # De-standardize the ground truth data
        gt_data_dict['groundTruth'] = unstandardize_predicted_features(gt_data_dict['groundTruth'])
        
        # Load, process, and de-standardize all prediction datasets
        predicted_datasets = []
        for pred_path in pred_file_paths:
            # We use a dummy gt_file_path here because load_and_process_predictions requires it
            # and the alignment logic is handled internally.
            pred_data_dict = load_and_process_predictions(pred_path, gt_file_path, radius_threshold_meters)
            if pred_data_dict:
                # De-standardize each predicted dataset
                pred_data_dict['predicted'] = unstandardize_predicted_features(pred_data_dict['predicted'])
                predicted_datasets.append(pred_data_dict['predicted'])
        
        # Define the labels for each prediction model
        prediction_labels = ["Model 1", "Model 2", "Model 3", "Model 4"]
        
        # Now, plot the comparison of CNR predictions
        plot_cnr_subplots(gt_data_dict['groundTruth'], predicted_datasets, prediction_labels)
        
        print("\nPlotted CNR comparison for all requested models.")"""