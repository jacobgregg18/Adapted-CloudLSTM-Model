import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datetime import datetime, timedelta
import numpy as np
import natsort
import time
import pickle
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
from torch.amp import autocast

print("Cuda environment is: ")
print(torch.cuda.is_available())
print(torch.version.cuda)
print("End", flush=True)

def gather_neighbors(x, idx):
    """
    x:   [B, N, C]
    idx: [B, N, k]
    ->   [B, N, k, C]
    """
    B, N, C = x.shape
    if N == 0 or idx.shape[2] == 0: # Handle empty point clouds or empty indices
        return torch.empty(B, N, idx.shape[2], C, dtype=x.dtype, device=x.device)

    idx_flat = idx.view(B, N * idx.shape[-1])
    batch_indices = torch.arange(B, device = x.device, dtype=torch.long).unsqueeze(1).repeat(1, N * idx.shape[-1])
    gathered_flat = x[batch_indices, idx_flat]

    # Reshape the gathered tensor to the final output shape
    return gathered_flat.view(B, N, idx.shape[-1], C)

class CloudLSTMCell(nn.Module):
    def __init__(self, in_dim, hidden_dim, msg_dim, k):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.k = k

        self.feat_linear = nn.Linear(in_dim, hidden_dim)

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, msg_dim), nn.ReLU(),
            nn.Linear(msg_dim, msg_dim), nn.ReLU(),
        )

        self.lstm = nn.LSTMCell(msg_dim + hidden_dim, hidden_dim)

    def forward(self, feat_t, indices, h_t_minus_1=None, c_t_minus_1=None):
        B, N, _ = feat_t.shape

        if h_t_minus_1 is None:
            h_t_minus_1 = torch.zeros(B, N, self.hidden_dim, device=feat_t.device)
            c_t_minus_1 = torch.zeros(B, N, self.hidden_dim, device=feat_t.device)

        if self.k == 0:
            msg = torch.zeros(B, N, self.msg_dim, device=feat_t.device)
        else:
            # KNN and neighbor gathering
            h_neighbors = gather_neighbors(h_t_minus_1, indices)
            # Message passing
            h_t_minus_1_expanded = h_t_minus_1.unsqueeze(2).repeat(1, 1, self.k, 1)
            msg_input = torch.cat([h_t_minus_1_expanded, h_neighbors], dim=-1)
            msg = self.msg_mlp(msg_input).sum(dim=2)

        # LSTM cell update
        lstm_input = torch.cat([msg, self.feat_linear(feat_t)], dim=-1)
        h_t, c_t = self.lstm(lstm_input.view(B*N, -1), (h_t_minus_1.view(B*N, -1), c_t_minus_1.view(B*N, -1)))

        return h_t.view(B, N, -1), c_t.view(B, N, -1)

class CloudLSTMNextScan(nn.Module):
    """
    Input:   sequence of T frames, each [N,8] (xyz + CNR + wind(3) + conf)
             shaped as [B,T,N,8]
    Output:  predicted next full feature vector [B,N,8] (xyz + CNR + wind(3) + conf at t+1)
             where xyz are copied from the last input frame.
    """
    def __init__(self, in_dim=8, hidden_dim=128, msg_dim=64, k=16, T=6):
        super().__init__()
        self.T = T
        self.in_dim = in_dim
        self.cell = CloudLSTMCell(in_dim=in_dim, hidden_dim=hidden_dim, msg_dim=msg_dim, k=k)

        # Number of features to predict (non-xyz)
        self.pred_dim = 1

        # The Prediction head 
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, self.pred_dim), # Outputs the predicted mean (delta features)
        )

    def forward(self, x, precomputed_indices):  # x: [B,T,N,F]
        B, T, N, F = x.shape
        assert T >= 2, "Need at least two frames (t-1, t)."

        # Handle empty point clouds in the batch
        if N == 0:
            return (torch.empty(B, N, self.in_dim, device=x.device, dtype=x.dtype),
                    torch.empty(B, N, self.pred_dim, device=x.device, dtype=x.dtype),
                    torch.empty(B, N, self.pred_dim, device=x.device, dtype=x.dtype))

        h = c = None
        # Unroll over time (uses per-frame xyz for neighborhoods)
        for t in range(T):
            feat_t = x[:, t, :, :]
            h, c = self.cell(feat_t, precomputed_indices, h, c)

        # Predict the delta CNR
        pred_delta_features = self.mean_head(h)  # [B,N,1]

        # Get the last full feature vector (xyz + additional features)
        last_features = x[:, -1, :, :7]

        # Get the constant coordinates from the last frame
        last_xyz = last_features[:, :, :3] # [B,N,3]

        # Get the additional features from the last frame
        last_additional_features = last_features[:, :, 3:4]
        last_wind_features = last_features[:, :, 4:7]

        # Predict the next additional feature vector by adding the delta
        pred_next_additional_feat = last_additional_features + pred_delta_features

        # Concatenate the constant's with the new predicted CNR
        pred_next_feat = torch.cat([last_xyz, pred_next_additional_feat, last_wind_features], dim=-1)

        return pred_next_feat, pred_delta_features

def min_max_scale(tensor, min_val, max_val):
    if tensor.numel() == 0:
        return tensor

    # Avoid division by zero if min_val and max_val are equal
    if max_val == min_val:
        return torch.zeros_like(tensor)
    else:
        scaled_tensor = (tensor - min_val) / (max_val - min_val)
        return scaled_tensor

def standardize(tensor, mean, std):
    if tensor.numel() == 0:
        return tensor

    # Avoid division by zero if std is zero
    if std == 0:
        return torch.zeros_like(tensor)
    else:
        standardized_tensor = (tensor - mean) / std
        return standardized_tensor

def log_normalize_cnr(cnr_tensor, cnr_shift, cnr_log_mean, cnr_log_std):
    """Apply log transformation and normalization to CNR values"""
    if cnr_tensor.numel() == 0:
        return cnr_tensor

    # Shift CNR to positive range
    cnr_shifted = cnr_tensor + cnr_shift
    # Safety clamp to avoid log(0) or log(negative)
    cnr_shifted = torch.clamp(cnr_shifted, min=1e-6)
    # Log transform
    cnr_log = torch.log(cnr_shifted)
    # Normalize using global log statistics
    cnr_normalized = (cnr_log - cnr_log_mean) / cnr_log_std

    return cnr_normalized

def normalize_winds_with_nonzero_stats(wind_values, scale):
    threshold = 1e-6
    zero_mask = torch.abs(wind_values) < threshold

    # Use p95 as scale (or std, your choice)
    normalized = wind_values / scale

    # Keep zeros as zeros
    normalized[zero_mask] = 0.0

    return normalized

class LTSMDataset(Dataset):
    def __init__(self, root_dir, seq_list, T=6, feature_stats=None, use_polar=False):
        self.root_dir = root_dir
        self.pc_dir = os.path.join(root_dir, 'diff_clouds_polar' if use_polar else 'diff_clouds')
        self.pc_files = natsort.natsorted(os.listdir(self.pc_dir))
        self.seq_list = seq_list
        self.T = T
        self.use_polar = use_polar
        self.max_radial_distance = 1.2

        # Keep your existing hardcoded stats
        self.hardcoded_min_stats = [-0.9845121092520055, -0.9845031637333835, 0.001796004840885169, -100.0, -2.2024194711095904, -1.8266775445970527, -2.3839595067237744, 1753711807.471]
        self.hardcoded_max_stats = [0.9845165259173817, 0.9845031637333835, 0.988206682848428, 91.48, 2.2221496199313333, 1.883747076787428, 2.025801534327347, 1754488505.001]
        self.hardcoded_mean_stats = [2.7022033600688215e-06, 4.1019552380086045e-06, 0.3304346729575734, -30.466586771026023, -0.012333, -0.005744, -0.002197, 1754092671.0763593]
        self.hardcoded_std_stats = [0.291664816512239, 0.2916651588805414, 0.24687948625791928, 4.496726643996928, 0.040166, 0.033383, 0.044368, 220765.19726839956]

        self.feature_stats = {
            'min': self.hardcoded_min_stats,
            'max': self.hardcoded_max_stats,
            'mean': self.hardcoded_mean_stats,
            'std': self.hardcoded_std_stats,
            'wind_scale': [0.15, 0.15, 0.15]
        }

    def __len__(self):
        return len(self.seq_list)

    def load_point_cloud(self, dt):
        if self.use_polar:
            fname = f"{dt.strftime('%Y-%m-%d')}_{dt.hour}_{dt.strftime('%M')}_polar.npy"
        else:
            fname = f"{dt.strftime('%Y-%m-%d')}_{dt.hour}_{dt.strftime('%M')}.npy"

        path = os.path.join(self.pc_dir, fname)
        if not os.path.exists(path):
            print(f"Missing point cloud: {path}")
            return None

        pc = np.load(path)
        if np.isnan(pc).any():
            print(f'Nan in file: {fname}')
            pc = np.nan_to_num(pc)

        # Apply radial distance filter
        if self.use_polar:
            # For polar: range is in column 1
            pc = pc[pc[:, 1] <= self.max_radial_distance * 14500]
        else:
            # For Cartesian: compute radial distance
            radial_distance = np.sqrt(pc[:, 0]**2 + pc[:, 1]**2 + pc[:, 2]**2)
            pc = pc[radial_distance <= self.max_radial_distance]

        return pc

    def standardize_features(self, pc_tensor):
        if self.use_polar:
            # Polar coordinate normalization
            # Azimuth: [-180, 180] -> [-1, 1]
            pc_tensor[:, 0] = pc_tensor[:, 0] / 180
            # Range: [0, 14500] -> [0, 1]
            pc_tensor[:, 1] = pc_tensor[:, 1] / 14500
            # Elevation: [-90, 90] -> [-1, 1]
            pc_tensor[:, 2] = pc_tensor[:, 2] / 90
            # CNR: Apply your supervisor's [-20, 20] constraint
            cnr_clipped = torch.clamp(pc_tensor[:, 3], -40, 10)
            pc_tensor[:, 3] = (cnr_clipped + 40) / 50  # [0, 1]
            pc_tensor = pc_tensor[:, :-1]
        else:
            pc_tensor = pc_tensor[:, :-1]
            for i in range(3):
                pc_tensor[:, i] = (pc_tensor[:, i] - self.feature_stats['mean'][i]) / self.feature_stats['std'][i]
            cnr_clipped = torch.clamp(pc_tensor[:, 3], -40, 10)
            pc_tensor[:, 3] = (cnr_clipped + 40) / 50  # [0, 1]

        return pc_tensor

    def downsample_point_cloud(self, pc, target_points):
        
        N = pc.shape[0]
        if N < target_points:
            pad = np.zeros((target_points-N, pc.shape[1]), dtype=np.float32)
            pc = np.vstack([pc, pad])
        elif N > target_points:
            idx = np.random.choice(N, target_points, replace=False)
            pc = pc[idx]
        return torch.from_numpy(pc).float()

    def __getitem__(self, idx):
        times = self.seq_list[idx]
        pcs = []
        for i in range(self.T+1):
            pc = self.load_point_cloud(times[i])
            if pc is None:
                return None
            pcs.append(pc)

        max_points = max(pc.shape[0] for pc in pcs)
        pcs_tensor = []
        for pc in pcs:
            pc_tensor = self.downsample_point_cloud(pc, max_points)
            pc_tensor = self.standardize_features(pc_tensor)
            pcs_tensor.append(pc_tensor)

        return {f'pc{i}': pc for i, pc in enumerate(pcs_tensor)}

    def collate_fn(self, batch):
        # Keep your existing logic
        batch = [b for b in batch if b is not None]
        if not batch: return {}
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}

class WeatherEnhancedLTSMDataset(LTSMDataset):
    def __init__(self, root_dir, seq_list, weather_csv_path, T=6, use_polar=False):
        super().__init__(root_dir, seq_list, T, feature_stats=None, use_polar=use_polar)

        # Weather features to use
        self.weather_features = [
            'TAAVG1M',      # Temperature
            'RHAVG1M',      # Humidity
            'TDAVG1M',      # Dew point
            'WS1AVG10M',    # Wind speed
            'WD1AVG10M',    # Wind direction
            'PRSUM1H',      # Precipitation
        ]

        # Load only required columns
        columns_to_load = ['datetime'] + self.weather_features
        self.weather_df = pd.read_csv(
            weather_csv_path,
            usecols=columns_to_load,
            low_memory=False
        )
        self.weather_df['datetime'] = pd.to_datetime(self.weather_df['datetime'])

        # Normalization ranges for weather
        self.weather_norm = {
            'TAAVG1M': (-10.0, 40.0),
            'RHAVG1M': (0.0, 100.0),
            'TDAVG1M': (-20.0, 30.0),
            'WS1AVG10M': (0.0, 30.0),
            'WD1AVG10M': (0.0, 360.0),
            'PRSUM1H': (0.0, 50.0),
        }

        if use_polar:
            self.grad_dir = os.path.join(root_dir, 'gradients_polar')
        else:
            self.grad_dir = os.path.join(root_dir, 'gradients')
        
        # Check if gradients exist
        self.has_precomputed_gradients = os.path.exists(self.grad_dir)
        
        if not self.has_precomputed_gradients:
            print("WARNING: No precomputed gradients found. Will compute on-the-fly (slow!)")
            print(f"Run precompute_gradients.py to create {self.grad_dir}")

    def verify_geometric_consistency(self):
        """Check if all point clouds have consistent geometry"""
        sample_scans = []
        for i in range(min(10, len(self.seq_list))):
            pc = self.load_point_cloud(self.seq_list[i][0])
            if pc is not None:
                # Get azimuth, range, elevation
                coords = pc[:, :3]
                sample_scans.append(coords)

        # Check if all scans have same shape and similar coordinate distributions
        for i, scan in enumerate(sample_scans):
            print(f"Scan {i}: shape={scan.shape}, "
                  f"az range=[{scan[:,0].min():.1f}, {scan[:,0].max():.1f}], "
                  f"range range=[{scan[:,1].min():.1f}, {scan[:,1].max():.1f}]")

    def get_weather_at_time(self, target_datetime, max_time_gap_minutes=15):
        """Get interpolated weather data for a specific datetime"""
        before = self.weather_df[self.weather_df['datetime'] <= target_datetime]
        after = self.weather_df[self.weather_df['datetime'] >= target_datetime]

        if len(before) == 0 or len(after) == 0:
            # Check if nearest data is within acceptable range
            time_diffs = abs(self.weather_df['datetime'] - target_datetime)
            min_diff = time_diffs.min()

            # If closest weather data is more than max_time_gap away, return None
            if min_diff > pd.Timedelta(minutes=max_time_gap_minutes):
                return None

            closest_idx = time_diffs.argmin()
            return self.weather_df.iloc[closest_idx]

        before_record = before.iloc[-1]
        after_record = after.iloc[0]

        # Check if the gap is too large
        before_gap = abs((target_datetime - before_record['datetime']).total_seconds() / 60)
        after_gap = abs((after_record['datetime'] - target_datetime).total_seconds() / 60)

        if before_gap > max_time_gap_minutes and after_gap > max_time_gap_minutes:
            return None

        if before_record['datetime'] == target_datetime:
            return before_record

        # Linear interpolation
        total_seconds = (after_record['datetime'] - before_record['datetime']).total_seconds()
        if total_seconds == 0:
            return before_record

        elapsed_seconds = (target_datetime - before_record['datetime']).total_seconds()
        weight = elapsed_seconds / total_seconds

        result = {}
        for feature in self.weather_features:
            before_val = before_record.get(feature, None)
            after_val = after_record.get(feature, None)

            if pd.isna(before_val) or pd.isna(after_val) or before_val is None or after_val is None:
                # If any required feature is missing, return None for the whole record
                return None

            if feature == 'WD1AVG10M':
                # Circular interpolation for wind direction
                diff = (after_val - before_val + 180) % 360 - 180
                result[feature] = (before_val + diff * weight) % 360
            else:
                result[feature] = before_val * (1 - weight) + after_val * weight

        return result

    def normalize_weather(self, feature_name, value):
        """Normalize weather feature to [0, 1]"""
        if feature_name not in self.weather_norm:
            return value

        min_val, max_val = self.weather_norm[feature_name]
        value = np.clip(value, min_val, max_val)
        return (value - min_val) / (max_val - min_val)

    def get_weather_features_normalized(self, target_datetime):
        """Get normalized weather features as numpy array, returns None if unavailable"""
        weather_data = self.get_weather_at_time(target_datetime)

        # Return None if no weather data available
        if weather_data is None:
            return None

        features = []
        for feature_name in self.weather_features:
            value = weather_data.get(feature_name, None)

            # If any feature is missing, return None
            if value is None or pd.isna(value):
                return None

            normalized = self.normalize_weather(feature_name, value)
            features.append(normalized)

        return np.array(features, dtype=np.float32)

    def compute_upwind_cnr_polar(self, pc_tensor, precomputed_indices, dt=600):
        """
        Compute upwind CNR using pre-computed neighbor indices

        pc_tensor: [N, 7] - [azimuth_deg, range_m, elevation_deg, cnr, wind_r, wind_t, wind_v]
        precomputed_indices: [N, k] - pre-computed k nearest neighbors for each point
        """
        N = pc_tensor.shape[0]
        k = precomputed_indices.shape[1]

        # Ensure indices are on the same device as pc_tensor
        precomputed_indices = precomputed_indices.to(pc_tensor.device)

        azimuth = pc_tensor[:, 0]
        range_m = pc_tensor[:, 1]
        elevation = pc_tensor[:, 2]
        current_cnr = pc_tensor[:, 3]

        wind_radial = pc_tensor[:, 4]
        wind_tangential = pc_tensor[:, 5]
        wind_vertical = pc_tensor[:, 6]

        # Convert to radians
        az_rad = torch.deg2rad(azimuth)
        el_rad = torch.deg2rad(elevation)

        # Polar to Cartesian
        x = range_m * torch.cos(el_rad) * torch.cos(az_rad)
        y = range_m * torch.cos(el_rad) * torch.sin(az_rad)
        z = range_m * torch.sin(el_rad)

        # Wind displacement components
        dx_radial = wind_radial * torch.cos(el_rad) * torch.cos(az_rad)
        dy_radial = wind_radial * torch.cos(el_rad) * torch.sin(az_rad)
        dz_radial = wind_radial * torch.sin(el_rad)

        dx_tangential = -wind_tangential * torch.sin(az_rad)
        dy_tangential = wind_tangential * torch.cos(az_rad)

        # Source positions (where air came from)
        source_x = x - (dx_radial + dx_tangential) * dt
        source_y = y - (dy_radial + dy_tangential) * dt
        source_z = z - (dz_radial + wind_vertical) * dt

        current_positions = torch.stack([x, y, z], dim=1)
        source_positions = torch.stack([source_x, source_y, source_z], dim=1)

        # Gather neighbor
        neighbor_positions = current_positions[precomputed_indices]

        # Compute distances from source
        source_expanded = source_positions.unsqueeze(1)
        distances = torch.norm(neighbor_positions - source_expanded, dim=2)

        # Find closest among pre-computed neighbors
        closest_idx = distances.argmin(dim=1)

        # Map back to actual point indices
        point_idx = torch.arange(N, device=pc_tensor.device)
        upwind_point_idx = precomputed_indices[point_idx, closest_idx]

        # Get upwind CNR
        upwind_cnr = current_cnr[upwind_point_idx]
        advection_delta = upwind_cnr - current_cnr

        return upwind_cnr, advection_delta
    
    def compute_radial_gradient_polar(self, pc_tensor):
        """
        Compute absolute CNR change along radial direction
        
        Returns:
            radial_grad: [N] - Absolute CNR change (range: [-1, 1])
            beam_cnr_std: [N] - CNR std within beam
            boundary_proximity: [N] - Normalized distance to strong gradients
        """
        N = pc_tensor.shape[0]
        device = pc_tensor.device
        
        # Extract features
        azimuth, range_m, elevation, cnr = pc_tensor[:, 0], pc_tensor[:, 1], pc_tensor[:, 2], pc_tensor[:, 3]
        
        # Create beam IDs
        az_bins = torch.round(azimuth).long()
        el_bins = torch.round(elevation).long()
        beam_id = (az_bins + 360) * 1000 + (el_bins + 90)
        
        # Initialize outputs
        radial_grad = torch.zeros(N, device=device)
        beam_cnr_std = torch.zeros(N, device=device)
        boundary_proximity = torch.ones(N, device=device) * 10.0
        
        # Process each beam
        for beam in torch.unique(beam_id):
            mask = beam_id == beam
            indices = torch.where(mask)[0]
            
            if len(indices) < 2:
                continue
            
            # Sort by range
            beam_ranges = range_m[indices]
            beam_cnrs = cnr[indices]
            sorted_idx = torch.argsort(beam_ranges)
            
            # Sorted data
            sorted_cnrs = beam_cnrs[sorted_idx]
            sorted_ranges = beam_ranges[sorted_idx]
            orig_indices = indices[sorted_idx]
            
            cnr_changes = torch.diff(sorted_cnrs)
            
            radial_grad[orig_indices[:-1]] = cnr_changes
            radial_grad[orig_indices[-1]] = cnr_changes[-1] if len(cnr_changes) > 0 else 0.0
            
            beam_cnr_std[indices] = torch.std(beam_cnrs) if len(beam_cnrs) > 1 else 0.0
            
            strong_boundaries = torch.abs(cnr_changes) > 0.3
            if strong_boundaries.any():
                boundary_locs = sorted_ranges[:-1][strong_boundaries]
                median_spacing = torch.median(torch.diff(sorted_ranges))
                
                for i, idx in enumerate(orig_indices):
                    if len(boundary_locs) > 0:
                        dist = torch.min(torch.abs(boundary_locs - sorted_ranges[i]))
                        boundary_proximity[idx] = torch.clamp(dist / (median_spacing + 1e-6), 0.0, 10.0)
        
        return radial_grad.unsqueeze(1), beam_cnr_std.unsqueeze(1), boundary_proximity.unsqueeze(1)

    def load_radial_gradient(self, dt):
        """Load precomputed radial gradient for a timestamp"""
        if not self.has_precomputed_gradients:
            return None
        
        if self.use_polar:
            fname = f"{dt.strftime('%Y-%m-%d')}_{dt.hour}_{dt.strftime('%M')}_polar.npy"
        else:
            fname = f"{dt.strftime('%Y-%m-%d')}_{dt.hour}_{dt.strftime('%M')}.npy"
        
        grad_path = os.path.join(self.grad_dir, fname)
        
        if not os.path.exists(grad_path):
            print(f"Missing gradient file: {grad_path}")
            return None
        
        return np.load(grad_path)
    
    def __getitem__(self, idx):
        times = self.seq_list[idx]
        pcs = []

        # Load T+1 frames
        for i in range(self.T + 1):
            pc = self.load_point_cloud(times[i])
            if pc is None:
                return None
            pcs.append(pc)

        # Uniform size
        max_points = max(pc.shape[0] for pc in pcs)
        pcs_tensor = []

        for i, pc in enumerate(pcs):
            pc_tensor = self.downsample_point_cloud(pc, max_points)
            pc_tensor = self.standardize_features(pc_tensor)

            # Get indices
            if hasattr(self, 'precomputed_indices') and max_points in self.precomputed_indices:
                indices = self.precomputed_indices[max_points]
            else:
                indices = precomputed_indices_tensor[0, :max_points, :]

            # Compute upwind features
            if self.use_polar:
                upwind_cnr, advection_delta = self.compute_upwind_cnr_polar(
                    pc_tensor, indices
                )
                upwind_cnr_expanded = upwind_cnr.unsqueeze(1)
                advection_delta_expanded = advection_delta.unsqueeze(1)
                radial_grad_np = self.load_radial_gradient(times[i])
                
                if radial_grad_np is not None:
                    # Downsample/pad gradient to match point cloud
                    N_orig = radial_grad_np.shape[0]
                    if N_orig < max_points:
                        # Pad with zeros
                        pad = np.zeros(max_points - N_orig, dtype=np.float32)
                        radial_grad_np = np.concatenate([radial_grad_np, pad])
                    elif N_orig > max_points:
                        radial_grad_np = radial_grad_np[:max_points]
                    
                    radial_grad = torch.from_numpy(radial_grad_np).float().unsqueeze(1)
                else:
                    # Fallback: compute on-the-fly
                    radial_grad, _, _ = self.compute_radial_gradient_polar(pc_tensor)
            else:
                N = pc_tensor.shape[0]
                upwind_cnr_expanded = torch.zeros(N, 1, device=pc_tensor.device)
                advection_delta_expanded = torch.zeros(N, 1, device=pc_tensor.device)
                radial_grad = torch.zeros(N, 1, device=pc_tensor.device)

            # Concatenate features
            enhanced_pc = torch.cat([
                pc_tensor,
                upwind_cnr_expanded,
                advection_delta_expanded
            ], dim=1)

            pcs_tensor.append(enhanced_pc)

        return {f'pc{i}': pc for i, pc in enumerate(pcs_tensor)}

# Loss with no additional weightings
def adaptive_loss(pred_cnr, tgt_cnr, prev_cnr):
    """
    Loss that focuses on dynamic regions while maintaining overall accuracy
    """
    # 1. Base MSE for overall accuracy
    loss_mse = F.mse_loss(pred_cnr, tgt_cnr)
    
    # 2. Delta MSE for temporal consistency
    tgt_delta = tgt_cnr - prev_cnr
    pred_delta = pred_cnr - prev_cnr
    loss_delta = F.mse_loss(pred_delta, tgt_delta)
    
    # 3. Dynamic region emphasis (|Δ| > 0.05)
    dynamic_mask = torch.abs(tgt_delta) > 0.05
    if dynamic_mask.any():
        loss_dynamic = F.mse_loss(pred_cnr[dynamic_mask], tgt_cnr[dynamic_mask])
    else:
        loss_dynamic = torch.tensor(0.0, device=pred_cnr.device)
    
    # 4. Variance preservation
    loss_var = torch.abs(pred_cnr.std() - tgt_cnr.std())
    
    loss = 0.3 * loss_mse + 0.4 * loss_dynamic + 0.2 * loss_delta + 0.1 * loss_var
    
    return loss

# Loss with adaptive MSE weighting
def adaptive_loss_with_cnr_weighting(pred_cnr, tgt_cnr, prev_cnr, 
                                      cnr_weight_power=2.0, 
                                      cnr_weight_scale=3.0):
    """
    Adaptive loss with exponential weighting for high CNR values
    
    Args:
        cnr_weight_power: How aggressively to weight high CNR
        cnr_weight_scale: Maximum weight multiplier for highest CNR
    """
    # 1. Base MSE for overall accuracy
    loss_mse = F.mse_loss(pred_cnr, tgt_cnr)
    
    # 2. Delta MSE for temporal consistency
    tgt_delta = tgt_cnr - prev_cnr
    pred_delta = pred_cnr - prev_cnr
    loss_delta = F.mse_loss(pred_delta, tgt_delta)
    
    # 3. Dynamic region emphasis (|Δ| > 0.05)
    dynamic_mask = torch.abs(tgt_delta) > 0.05
    if dynamic_mask.any():
        loss_dynamic = F.mse_loss(pred_cnr[dynamic_mask], tgt_cnr[dynamic_mask])
    else:
        loss_dynamic = torch.tensor(0.0, device=pred_cnr.device)
    
    # 4. Variance preservation
    loss_var = torch.abs(pred_cnr.std() - tgt_cnr.std())
    
    # 5. Exponential weighting
    cnr_weights = 1.0 + (cnr_weight_scale - 1.0) * torch.pow(tgt_cnr, cnr_weight_power)
    weighted_error = cnr_weights * (pred_cnr - tgt_cnr) ** 2
    loss_cnr_weighted = weighted_error.mean()
    
    # Combined loss
    loss = (0.2 * loss_mse + 
            0.3 * loss_dynamic + 
            0.1 * loss_delta + 
            0.05 * loss_var +
            0.35 * loss_cnr_weighted)
    
    return loss

# Loss with adaptive and changing weighted Huber
class ImprovedCloudDynamicsLoss(nn.Module):
    """
    Combines best of Experiment 171 with improved weighting
    """
    def __init__(self, max_epochs=5, huber_delta=0.2, 
                 cloud_boost=1.5, dynamic_threshold=0.05):
        super().__init__()
        self.max_epochs = max_epochs
        self.current_epoch = 0
        self.huber_delta = huber_delta
        self.cloud_boost = cloud_boost
        self.dynamic_threshold = dynamic_threshold
        
        # Huber loss for robustness
        self.huber = nn.HuberLoss(reduction='none', delta=huber_delta)
    
    def forward(self, pred_cnr, tgt_cnr, prev_cnr, radial_grad=None):
        """
        Compute multi-component cloud-aware loss
        """
        
        # Compute deltas
        pred_delta = pred_cnr - prev_cnr
        tgt_delta = tgt_cnr - prev_cnr
        
        # 1: Base MSE (global accuracy)
        loss_mse = F.mse_loss(pred_cnr, tgt_cnr)
        
        # 2: Dynamic Region Emphasis (|Δ| > threshold)
        dynamic_mask = torch.abs(tgt_delta) > self.dynamic_threshold
        
        if dynamic_mask.any():
            loss_dynamic = F.mse_loss(
                pred_cnr[dynamic_mask], 
                tgt_cnr[dynamic_mask]
            )
        else:
            loss_dynamic = torch.tensor(0.0, device=pred_cnr.device)
        
        # 3: Temporal Delta Consistency
        loss_delta = F.mse_loss(pred_delta, tgt_delta)
        
        # 4: Variance Preservation (prevent collapse)
        pred_std = pred_cnr.std()
        tgt_std = tgt_cnr.std()
        loss_var = torch.abs(pred_std - tgt_std)
        
        # 5: Weighted Huber Loss
        # Progressive scheduling
        progress = min(self.current_epoch / self.max_epochs, 1.0)
        
        # Squeeze for weight computation
        cnr = prev_cnr.squeeze(-1)
        target_d = tgt_delta.squeeze(-1)
        
        # Dynamic magnitude weight -> Emphasize large changes progressively
        beta = 1.5 + 0.5 * progress
        dynamic_weight = torch.pow(torch.abs(target_d), beta)
        
        # Normalize to [0, 1]
        if dynamic_weight.max() > 0:
            dynamic_weight = dynamic_weight / dynamic_weight.max()
        else:
            dynamic_weight = torch.zeros_like(dynamic_weight)
        
        # Cloud region weight -> Moderate boost for high-CNR regions
        cloud_threshold = 0.4
        cloud_mask_w = (cnr > cloud_threshold).float()
        cloud_excess = torch.clamp(cnr - cloud_threshold, min=0)
        cloud_weight = 1.0 + self.cloud_boost * cloud_excess * cloud_mask_w
        
        # Boundary weight
        if radial_grad is not None:
            grad = radial_grad.squeeze(-1)
            is_boundary = (torch.abs(grad) > 0.3).float()
            boundary_weight = 1.0 + 0.5 * is_boundary
        else:
            boundary_weight = 1.0
        
        # Combine weights
        alpha = 0.4 - 0.2 * progress  # 0.4 → 0.2
        
        spatial_weight = cnr * cloud_weight * boundary_weight
        combined_weight = alpha * spatial_weight + (1 - alpha) * dynamic_weight
        combined_weight = torch.clamp(combined_weight, min=0.3, max=2.0)
        
        # Apply to Huber loss
        combined_weight = combined_weight.unsqueeze(-1)  # [B, N, 1]
        loss_pointwise = self.huber(pred_delta, tgt_delta)
        weighted_loss = loss_pointwise * combined_weight
        loss_weighted_huber = weighted_loss.sum() / (combined_weight.sum() + 1e-6)
        
        loss = (
            0.15 * loss_mse +
            0.30 * loss_dynamic +
            0.15 * loss_delta +
            0.05 * loss_var +
            0.35 * loss_weighted_huber
        )
        
        # Loss components for logging
        loss_dict = {
            'total': loss.item(),
            'mse': loss_mse.item(),
            'dynamic': loss_dynamic.item() if isinstance(loss_dynamic, torch.Tensor) else 0.0,
            'delta': loss_delta.item(),
            'var': loss_var.item(),
            'weighted_huber': loss_weighted_huber.item(),
            'weight_mean': combined_weight.mean().item(),
            'weight_max': combined_weight.max().item(),
        }
        
        return loss, loss_dict
    
    def step_epoch(self):
        """Call at end of each epoch to update progressive scheduling"""
        self.current_epoch += 1
        print(f"  → Loss scheduler: Epoch {self.current_epoch}/{self.max_epochs}, "
              f"Progress: {self.current_epoch/self.max_epochs:.2%}")



## Start Run ##
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lidar_dir = os.getcwd()
weather_csv = "weather_data_combined.csv"
T=6

# Create sequence list from filenames
files = sorted(os.listdir(os.path.join(lidar_dir, "diff_clouds")))
datetimes = [datetime.strptime(f.split(".")[0], "%Y-%m-%d_%H_%M") for f in files]
seqs = [datetimes[i:i+T+1] for i in range(len(datetimes)-T)]
split = int(0.85 * len(seqs))
train_seqs = seqs[:split]
val_seqs = seqs[split:]

train_dataset = WeatherEnhancedLTSMDataset(lidar_dir, train_seqs, weather_csv, T=T, use_polar=True)
val_dataset = WeatherEnhancedLTSMDataset(lidar_dir, val_seqs, weather_csv, T=T, use_polar=True)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)

model = CloudLSTMNextScan(in_dim=9, hidden_dim=384, msg_dim=96, k=0, T=T).to(device, dtype=torch.float32)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scaler = GradScaler()

knn_file = os.path.join(lidar_dir, "precomputed_neighbors/K32.pt")
if os.path.exists(knn_file):
  print("Found file")
precomputed_indices_tensor = torch.load(knn_file, pickle_module=pickle)
precomputed_indices_tensor = precomputed_indices_tensor.to(device=device, dtype=torch.long)

num_epochs = 10

# Optionally load from checkpoint
start_epoch = 0
base_path = "Test7_LSTM_Epoch"
checkpoint_path = "Test7_LSTM.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=pickle)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Resumed from checkpoint")

torch.cuda.empty_cache()
starttime = time.time()
criterion = ImprovedCloudDynamicsLoss(
        max_epochs=num_epochs,
        huber_delta=0.2,
        cloud_boost=1.5,
        dynamic_threshold=0.05
    )

print(f"Control Group pass {checkpoint_path}")

for epoch in range(num_epochs):
    model.train()
    total_loss = total_empirical_bias = total_pred_change = 0.0
    total_target_change = total_l1_loss = total_cloud_loss = total_cloud_loss1 = 0.0
    total_bg_loss = total_bg_loss1 = pct_active = 0.0
    total_l1_per = total_mse_loss = total_mse_per = 0.0
    y = 0
    epoch_components = {
            'mse': 0.0,
            'dynamic': 0.0,
            'delta': 0.0,
            'var': 0.0,
            'weighted_huber': 0.0
        }
    
    for batch in train_loader:
        # Check if batch dictionary is empty or missing expected keys
        if not batch:
            print("Skipping empty batch.")
            continue

        # Get all point cloud keys dynamically
        pc_keys = [key for key in batch.keys() if key.startswith('pc')]
        pc_keys.sort()  # Ensure correct order

        if len(pc_keys) < T + 1:  # Need T inputs + 1 target
            print("Insufficient point clouds in batch.")
            continue

        # Load point clouds to device
        point_clouds = []
        skip_batch = False

        for key in pc_keys:
            pc = batch[key].to(device)
            if pc.size(1) == 0:  # Check for empty point clouds
                print(f"Empty point cloud found: {key}, skipping batch")
                skip_batch = True
                break
            point_clouds.append(pc)

        if skip_batch:
            y += 1
            continue

        # Split into inputs and target
        input_pcs = point_clouds[:T]  # First T point clouds for input
        target_pc = point_clouds[T]   # Last point cloud for target
        # Stack the input point clouds: [B, T, N, F]
        x = torch.stack(input_pcs, dim=1)

        pc_current = point_clouds[T-1]
        pc_target = target_pc

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', enabled=(device.type=='cuda')):
            pred_next_feat, pred_delta = model(x, precomputed_indices_tensor)

            if torch.isnan(pred_next_feat).any() or torch.isnan(pred_delta).any():
                print("Nan in model output")
                continue

            # Only consider CNR channel
            prev_cnr = pc_current[:, :, 3:4]
            tgt_delta = pc_target[:, :, 3:4] - prev_cnr
            pred_cnr = pred_next_feat[:, :, 3:4]
            tgt_cnr = pc_target[:, :, 3:4]
            radial_grad = None
            
            # Weighted loss
            loss, loss_dict = criterion(
                pred_cnr=pred_cnr,
                tgt_cnr=tgt_cnr,
                prev_cnr=prev_cnr,
                radial_grad=radial_grad
            )

            if torch.isnan(loss):
                print("NaN loss; skipping step.")
                continue
        
        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for key in epoch_components:
            epoch_components[key] += loss_dict[key]

        # Metrics
        with torch.no_grad():
            # Global L1 loss
            total_l1_loss += F.l1_loss(pred_cnr, tgt_cnr, reduction='mean')
            total_l1_per += F.l1_loss(prev_cnr, tgt_cnr, reduction='mean')

            # Global MSE loss
            total_mse_loss += F.mse_loss(pred_cnr, tgt_cnr, reduction='mean')
            total_mse_per += F.mse_loss(prev_cnr, tgt_cnr, reduction='mean')

            # Changes relative to current
            pred_change   = pred_cnr - prev_cnr
            actual_change = tgt_cnr - prev_cnr
            total_pred_change   += torch.abs(pred_change).mean()
            total_target_change += torch.abs(actual_change).mean()

            # Empirical bias
            total_empirical_bias += (pred_delta.mean() - tgt_delta.mean()).item()

            # Cloud vs. background split
            cloud_mask = prev_cnr > 0.4
            bg_mask    = prev_cnr <= 0.4

            if cloud_mask.any():
                cloud_loss = F.l1_loss(pred_cnr[cloud_mask], tgt_cnr[cloud_mask])
                total_cloud_loss += cloud_loss.item()

            if bg_mask.any():
                bg_loss = F.l1_loss(pred_cnr[bg_mask], tgt_cnr[bg_mask])
                total_bg_loss += bg_loss.item()
            
            cloud_mask = tgt_cnr > 0.4
            bg_mask    = tgt_cnr <= 0.4

            if cloud_mask.any():
                cloud_loss = F.l1_loss(pred_cnr[cloud_mask], tgt_cnr[cloud_mask])
                total_cloud_loss1 += cloud_loss.item()

            if bg_mask.any():
                bg_loss = F.l1_loss(pred_cnr[bg_mask], tgt_cnr[bg_mask])
                total_bg_loss1 += bg_loss.item()

        # Debug Prints
        if y % 400 == 0:
            print(y)
            with torch.no_grad():
                p = pred_delta.view(-1).cpu()
                t = tgt_delta.view(-1).cpu()

                def show_stats(name, arr):
                    arr = arr[torch.isfinite(arr)]
                    if arr.numel() == 0:
                        print(name, "empty")
                        return
                    print(f"{name}: mean={arr.mean():.4f}, std={arr.std():.4f}, "
                          f"p50={arr.median():.4f}, p90={arr.kthvalue(int(0.9*arr.numel()))[0]:.4f}, p95={arr.kthvalue(int(0.95*arr.numel()))[0]:.4f}")

                show_stats("pred_delta_std", p)
                show_stats("tgt_delta_std", t)
                print(f"Target Delta stats: Max: {tgt_delta.max()} | Min {tgt_delta.min()} | Mean: {tgt_delta.mean()}")
                print(f"Predicted Delta stats: Max: {pred_delta.max()} | Min {pred_delta.min()} | Mean: {pred_delta.mean()}")
                print("pct_nonzero_tgt (1dB+):", (t.abs() > 0.02).float().mean().item(), flush=True)

            torch.cuda.empty_cache()
        
        if y % 5 == 0:
            torch.cuda.empty_cache()

        del x, pred_next_feat, pred_delta, loss
        y += 1

    # Epoch Summary
    n_batches = max(1, len(train_loader))
    avg_loss = total_loss / n_batches
    print(f"[Epoch {epoch+1}] Train Average Total Loss: {total_loss / n_batches:.6f}", flush=True)
    print(f" Empirical Bias: {total_empirical_bias / n_batches}")
    print(f" Pred Change: {total_pred_change / n_batches} | Target Change: {total_target_change / n_batches}")
    print(f" L1 Loss: {total_l1_loss / n_batches}, Persistance L1: {total_l1_per / n_batches}")
    print(f" MSE Loss: {total_mse_loss / n_batches}, Persistance MSE: {total_mse_per / n_batches}")
    print(f" Cloud Loss (prev cnr > 0.4): {total_cloud_loss / n_batches} | Background Loss (prev cnr < 0.4): {total_bg_loss / n_batches}")
    print(f" Cloud Loss (tgt cnr > 0.4): {total_cloud_loss1 / n_batches} | Background Loss (tgt cnr < 0.4): {total_bg_loss1 / n_batches}", flush=True)
    print(f"    - MSE (15%):           {epoch_components['mse']/n_batches:.6f}")
    print(f"    - Dynamic (30%):       {epoch_components['dynamic']/n_batches:.6f}")
    print(f"    - Delta (15%):         {epoch_components['delta']/n_batches:.6f}")
    print(f"    - Variance (5%):       {epoch_components['var']/n_batches:.6f}")
    print(f"    - Weighted Huber (35%): {epoch_components['weighted_huber']/n_batches:.6f}")
    criterion.step_epoch()
    
    # Run Validation Set
    model.eval()
    results = {
        'easy': {'model_mse': [], 'persist_mse': [], 'model_mae': [], 'persist_mae': [], 'corr': []},
        'medium': {'model_mse': [], 'persist_mse': [], 'model_mae': [], 'persist_mae': [], 'corr': []},
        'hard': {'model_mse': [], 'persist_mse': [], 'model_mae': [], 'persist_mae': [], 'corr': []},
        'dynamic_region': {'model_mse': [], 'persist_mse': [], 'model_mae': [], 'persist_mae': []}
    }
    
    with torch.no_grad():
        for batch in val_loader:
            # Get point clouds
            pc_keys = [key for key in batch.keys() if key.startswith('pc')]
            pc_keys.sort()
            
            if len(pc_keys) < T:
                continue
                
            point_clouds = []
            for key in pc_keys:
                pc = batch[key].to(device)
                if pc.size(1) == 0:
                    break
                point_clouds.append(pc)
            
            if len(point_clouds) < T+1:
                continue
            
            # Prepare inputs
            input_pcs = point_clouds[:T]
            x = torch.stack(input_pcs, dim=1)
            
            pc_current = point_clouds[T-1]
            pc_target = point_clouds[T]
            
            # Get predictions
            pred_next_feat, delta_cnr = model(x, precomputed_indices_tensor)
            pred_cnr = pred_next_feat[:, :, 3:4]
            tgt_cnr = pc_target[:, :, 3:4]
            prev_cnr = pc_current[:, :, 3:4]
            
            # Compute correlation (scene difficulty)
            corr = torch.corrcoef(torch.stack([
                prev_cnr.flatten(),
                tgt_cnr.flatten()
            ]))[0, 1].item()
            
            # Compute metrics
            model_mse = F.mse_loss(pred_cnr, tgt_cnr).item()
            persist_mse = F.mse_loss(prev_cnr, tgt_cnr).item()
            model_mae = F.l1_loss(pred_cnr, tgt_cnr).item()
            persist_mae = F.l1_loss(prev_cnr, tgt_cnr).item()
            
            # Categorize by difficulty
            if corr > 0.85:
                category = 'easy'
            elif corr > 0.65:
                category = 'medium'
            else:
                category = 'hard'
            
            results[category]['model_mse'].append(model_mse)
            results[category]['persist_mse'].append(persist_mse)
            results[category]['model_mae'].append(model_mae)
            results[category]['persist_mae'].append(persist_mae)
            results[category]['corr'].append(corr)
            
            # Evaluate on dynamic regions only
            tgt_delta = tgt_cnr - prev_cnr
            dynamic_mask = torch.abs(tgt_delta) > 0.05
            
            if dynamic_mask.any():
                model_mse_dyn = F.mse_loss(pred_cnr[dynamic_mask], tgt_cnr[dynamic_mask]).item()
                persist_mse_dyn = F.mse_loss(prev_cnr[dynamic_mask], tgt_cnr[dynamic_mask]).item()
                model_mae_dyn = F.l1_loss(pred_cnr[dynamic_mask], tgt_cnr[dynamic_mask]).item()
                persist_mae_dyn = F.l1_loss(prev_cnr[dynamic_mask], tgt_cnr[dynamic_mask]).item()
                
                results['dynamic_region']['model_mse'].append(model_mse_dyn)
                results['dynamic_region']['persist_mse'].append(persist_mse_dyn)
                results['dynamic_region']['model_mae'].append(model_mae_dyn)
                results['dynamic_region']['persist_mae'].append(persist_mae_dyn)
    
    # Print summary
    print("\n" + "="*70)
    print("STRATIFIED EVALUATION RESULTS")
    print("="*70)
    
    for cat in ['easy', 'medium', 'hard']:
        if len(results[cat]['model_mse']) == 0:
            continue
            
        model_mse = np.mean(results[cat]['model_mse'])
        persist_mse = np.mean(results[cat]['persist_mse'])
        model_mae = np.mean(results[cat]['model_mae'])
        persist_mae = np.mean(results[cat]['persist_mae'])
        avg_corr = np.mean(results[cat]['corr'])
        
        mse_improvement = 100 * (persist_mse - model_mse) / persist_mse
        mae_improvement = 100 * (persist_mae - model_mae) / persist_mae
        
        print(f"\n{cat.upper()} SCENES (n={len(results[cat]['model_mse'])}, corr={avg_corr:.3f}):")
        print(f"  MSE  - Model: {model_mse:.6f} | Persist: {persist_mse:.6f} | Δ: {mse_improvement:+.1f}%")
        print(f"  MAE  - Model: {model_mae:.6f} | Persist: {persist_mae:.6f} | Δ: {mae_improvement:+.1f}%")
    
    # Dynamic region results
    if len(results['dynamic_region']['model_mse']) > 0:
        model_mse_dyn = np.mean(results['dynamic_region']['model_mse'])
        persist_mse_dyn = np.mean(results['dynamic_region']['persist_mse'])
        model_mae_dyn = np.mean(results['dynamic_region']['model_mae'])
        persist_mae_dyn = np.mean(results['dynamic_region']['persist_mae'])
        
        mse_improvement_dyn = 100 * (persist_mse_dyn - model_mse_dyn) / persist_mse_dyn
        mae_improvement_dyn = 100 * (persist_mae_dyn - model_mae_dyn) / persist_mae_dyn
        
        print(f"\nDYNAMIC REGIONS ONLY (|Δ| > 0.05, n={len(results['dynamic_region']['model_mse'])}):")
        print(f"  MSE  - Model: {model_mse_dyn:.6f} | Persist: {persist_mse_dyn:.6f} | Δ: {mse_improvement_dyn:+.1f}%")
        print(f"  MAE  - Model: {model_mae_dyn:.6f} | Persist: {persist_mae_dyn:.6f} | Δ: {mae_improvement_dyn:+.1f}%")
    
    print("="*70 + "\n", flush=True)

    # save overall model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss
    }, checkpoint_path)

    # Save the epoch specific model
    path = base_path + f"{epoch}" + ".pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss
    }, path)

endtime = time.time()
print("Timing was:")
print(endtime - starttime)