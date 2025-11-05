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

def save(x, model):
    model.eval()
    with torch.no_grad():
        y = 0
        for batch in test_loader:
            # Check if batch dictionary is empty or missing expected keys
            if not batch:
                print("Skipping empty batch.")
                continue

            # Get all point cloud keys dynamically
            pc_keys = [key for key in batch.keys() if key.startswith('pc')]
            pc_keys.sort()  # Ensure correct order

            # Check if we have enough point clouds
            if len(pc_keys) < T + 1:
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
            input_pcs = point_clouds[:T]
            target_pc = point_clouds[T]
            x = torch.stack(input_pcs, dim=1)

            pc_current = point_clouds[T-1]
            pc_target = target_pc

            pred_next_feat, pred_delta = model(x, precomputed_indices_tensor)

            pred_np = pred_next_feat.detach().cpu().numpy().squeeze(axis=0)
            filename = f"LSTM_prediction{x}{y}.npy"
            output_filename = os.path.join(lidar_dir, filename)
            np.save(output_filename, pred_np)
            print(f"\nSaved validation prediction to: {output_filename}")
            y += 1

## Start Run ##
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lidar_dir = os.getcwd()

weather_csv = "weather_data_combined.csv"
T=6
# Create sequence list from filenames
files = sorted(os.listdir(os.path.join(lidar_dir, "diff_clouds")))
datetimes = [datetime.strptime(f.split(".")[0], "%Y-%m-%d_%H_%M") for f in files]
seqs = [datetimes[i:i+T+1] for i in range(len(datetimes)-T)]

test_seqs = [
    [datetime(2025,7,31,10,10), datetime(2025,7,31,10,10), datetime(2025,7,31,10,20), datetime(2025,7,31,10,30), datetime(2025,7,31,10,40), datetime(2025,7,31,10,50), datetime(2025,7,31,11,00),
     datetime(2025,7,31,11,10), datetime(2025,7,31,11,20), datetime(2025,7,31,11,30), datetime(2025,7,31,11,40),
     datetime(2025,7,31,11,50), datetime(2025,7,31,12,00)],

    [datetime(2025,8,4,1,40), datetime(2025,8,4,1,50), datetime(2025,8,4,2,00), datetime(2025,8,4,2,10), datetime(2025,8,4,2,20), datetime(2025,8,4,2,30), datetime(2025,8,4,2,40),
     datetime(2025,8,4,2,50), datetime(2025,8,4,3,00), datetime(2025,8,4,3,10), datetime(2025,8,4,3,20),
     datetime(2025,8,4,3,30), datetime(2025,8,4,3,40)],

    [datetime(2025,8,2,19,30), datetime(2025,8,2,19,50), datetime(2025,8,2,20,00), datetime(2025,8,2,20,20), datetime(2025,8,2,20,30), datetime(2025,8,2,20,40), datetime(2025,8,2,21,00),
     datetime(2025,8,2,21,10), datetime(2025,8,2,21,30), datetime(2025,8,2,21,40), datetime(2025,8,2,21,50),
     datetime(2025,8,2,22,00), datetime(2025,8,2,22,10)],

    [datetime(2025,8,1,7,10), datetime(2025,8,1,7,20), datetime(2025,8,1,7,30), datetime(2025,8,1,7,40), datetime(2025,8,1,7,50), datetime(2025,8,1,8,00), datetime(2025,8,1,8,10),
     datetime(2025,8,1,8,20), datetime(2025,8,1,8,30), datetime(2025,8,1,8,40), datetime(2025,8,1,8,50),
     datetime(2025,8,1,9,00), datetime(2025,8,1,9,10)]
]
test_dataset = WeatherEnhancedLTSMDataset(lidar_dir, test_seqs, weather_csv, T=T, use_polar=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)


model = CloudLSTMNextScan(in_dim=9, hidden_dim=384, msg_dim=96, k=32, T=T).to(device, dtype=torch.float32) # Ensure model parameters are float32
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

knn_file = os.path.join(lidar_dir, "precomputed_neighbors/K32.pt")
if os.path.exists(knn_file):
  print("Found file")
precomputed_indices_tensor = torch.load(knn_file, pickle_module=pickle)
precomputed_indices_tensor = precomputed_indices_tensor.to(device=device, dtype=torch.long)

checkpoint_path = "Test_LSTM_Epoch0.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=pickle)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
save(1, model)

checkpoint_path = "Test_LSTM_Epoch1.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=pickle)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
save(2, model)

checkpoint_path = "Test_LSTM_Epoch2.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=pickle)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
save(3, model)

checkpoint_path = "Test_LSTM_Epoch3.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=pickle)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
save(4, model)

checkpoint_path = "Test_LSTM_Epoch4.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=pickle)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
save(5, model)

checkpoint_path = "Test_LSTM_Epoch5.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=pickle)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
save(6, model)

checkpoint_path = "Test_LSTM_Epoch6.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=pickle)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
save(7, model)

checkpoint_path = "Test_LSTM_Epoch7.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=pickle)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
save(8, model)

checkpoint_path = "Test_LSTM_Epoch8.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=pickle)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
save(9, model)

checkpoint_path = "Test_LSTM_Epoch9.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=pickle)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
save(10, model)
