import torch
from torch.utils.data import Dataset

import uproot
import awkward as ak
import numpy as np
import hashlib
import os
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

from glob import glob

class MinMaxScaler:
    def __init__(self, data):
        self.min = data.min(dim=0).values
        self.max = data.max(dim=0).values

    def transform(self, data):
        return (data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data):
        return data * (self.max - self.min + 1e-8) + self.min
    
    def __call__(self, sample):
        data, label = sample
        return self.transform(data), label


def _process_single_file(args):
    file_path, columns = args
    try:
        with uproot.open(file_path + ":trackingNtuple/tree") as tree:
            arrays = tree.arrays(columns + ["trk_simTrkIdx"], library="ak")
            
            features = {col: arrays[col] for col in columns}
            labels = ak.fill_none(ak.firsts(ak.flatten(arrays["trk_simTrkIdx"], axis=1)), -1) != -1
            
            return features, labels
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


class TrackDataset(Dataset):
    def __init__(self, input_files, transform=None, cache_dir=None, n_workers=None, batch_size=100, lazy_load=True):
        self.data = []
        self.labels = []
        self.lazy_load = lazy_load
        self.batch_cache_files = []
        self.batch_ranges = []
        self._loaded_batches = {}
        self._total_samples = 0
        
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / 'cache'
        else:
            cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        
        if n_workers is None:
            n_workers = cpu_count()
        self.n_workers = max(1, n_workers)
        self.batch_size = batch_size
        
        file_list = sorted(glob(input_files) if isinstance(input_files, str) else input_files)
        
        if len(file_list) == 0:
            raise ValueError(f"No files found matching pattern: {input_files}")
        
        print(f"Found {len(file_list)} files to process")
        
        if lazy_load:
            print("Using lazy loading mode (memory efficient for multi-GPU)")
            self._setup_lazy_loading(file_list, cache_dir)
        else:
            print("Using eager loading mode (loads all data into RAM)")
            self._process_and_load_batches(file_list, cache_dir)

        if transform is not None and isinstance(transform, type):
            if lazy_load:
                print("Computing transform parameters from first batch...")
                first_batch = torch.load(self.batch_cache_files[0], weights_only=False)
                sample_data = first_batch['data']
                self.transform = transform(sample_data)
                print(f"Transform initialized using {len(sample_data):,} samples from first batch")
            else:
                self.transform = transform(self.data)
        else:
            self.transform = transform
    
    def _generate_cache_key(self, file_list):
        file_info = []
        for file_path in file_list:
            if os.path.exists(file_path):
                mtime = os.path.getmtime(file_path)
                file_info.append(f"{file_path}:{mtime}")
        
        hash_str = '|'.join(file_info)
        return hashlib.md5(hash_str.encode()).hexdigest()[:16]
    
    def _get_batch_cache_file(self, cache_dir, batch_files):
        cache_key = self._generate_cache_key(batch_files)
        print(cache_dir)
        return cache_dir / f'batch_{cache_key}.pt'
    
    def _setup_lazy_loading(self, file_list, cache_dir):
        num_batches = (len(file_list) + self.batch_size - 1) // self.batch_size
        
        print(f"Setting up lazy loading for {len(file_list)} files in {num_batches} batches")
        print(f"Using {self.n_workers} worker(s) for parallel processing")
        
        current_idx = 0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(file_list))
            batch_files = file_list[start_idx:end_idx]
            
            batch_cache_file = self._get_batch_cache_file(cache_dir, batch_files)
            
            print(f"\n[Batch {batch_idx + 1}/{num_batches}] Files {start_idx + 1}-{end_idx}")
            
            if batch_cache_file.exists():
                print(f"  Found cache: {batch_cache_file.name}")
                try:
                    cached_batch = torch.load(batch_cache_file, weights_only=False)
                    batch_size_samples = len(cached_batch['data'])
                    print(f"  Contains {batch_size_samples:,} samples")
                except Exception as e:
                    print(f"  Warning: Failed to read cache ({e}), reprocessing...")
                    batch_cache_file.unlink()
                    _, _ = self._process_batch(batch_files, batch_cache_file)
                    cached_batch = torch.load(batch_cache_file, weights_only=False)
                    batch_size_samples = len(cached_batch['data'])
            else:
                print(f"  No cache found, processing batch...")
                _, _ = self._process_batch(batch_files, batch_cache_file)
                cached_batch = torch.load(batch_cache_file, weights_only=False)
                batch_size_samples = len(cached_batch['data'])
            
            self.batch_cache_files.append(batch_cache_file)
            batch_end_idx = current_idx + batch_size_samples
            self.batch_ranges.append((current_idx, batch_end_idx))
            current_idx = batch_end_idx
        
        self._total_samples = current_idx
        print(f"\nLazy loading setup complete: {self._total_samples:,} total samples across {num_batches} batches")
        print(f"Memory usage: Minimal (batches loaded on-demand)")
    
    def _load_batch_by_index(self, batch_idx):
        batch_cache_file = self.batch_cache_files[batch_idx]
        cached_batch = torch.load(batch_cache_file, weights_only=False)
        return {
            'data': cached_batch['data'],
            'labels': cached_batch['labels']
        }
    
    def _find_batch_for_idx(self, idx):
        for batch_idx, (start, end) in enumerate(self.batch_ranges):
            if start <= idx < end:
                return batch_idx, idx - start
        raise IndexError(f"Index {idx} out of range")
    
    def _process_and_load_batches(self, file_list, cache_dir):
        num_batches = (len(file_list) + self.batch_size - 1) // self.batch_size
        
        print(f"Processing {len(file_list)} files in {num_batches} batches of {self.batch_size} files each")
        print(f"Using {self.n_workers} worker(s) for parallel processing")
        
        all_data = []
        all_labels = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(file_list))
            batch_files = file_list[start_idx:end_idx]
            
            batch_cache_file = self._get_batch_cache_file(cache_dir, batch_files)
            
            print(f"\n[Batch {batch_idx + 1}/{num_batches}] Processing files {start_idx + 1}-{end_idx} of {len(file_list)}")
            
            # Try to load from cache
            if batch_cache_file.exists():
                print(f"  Loading from cache: {batch_cache_file.name}")
                try:
                    cached_batch = torch.load(batch_cache_file, weights_only=False)
                    batch_data = cached_batch['data']
                    batch_labels = cached_batch['labels']
                    print(f"  Loaded {len(batch_data):,} samples from cache")
                except Exception as e:
                    print(f"  Warning: Failed to load cache ({e}), reprocessing...")
                    batch_cache_file.unlink()
                    batch_data, batch_labels = self._process_batch(batch_files, batch_cache_file)
            else:
                print(f"  No cache found, processing batch...")
                batch_data, batch_labels = self._process_batch(batch_files, batch_cache_file)
            
            all_data.append(batch_data)
            all_labels.append(batch_labels)
        
        print(f"\nConcatenating {num_batches} batches...")
        self.data = torch.cat(all_data, dim=0)
        self.labels = torch.cat(all_labels, dim=0)
        
        print(f"Final dataset: {len(self.data):,} samples, {self.data.shape[1]} features")
    
    def _process_batch(self, batch_files, cache_file):
        start_time = time.time()
        
        columns = [
            'trk_px', 'trk_py', 'trk_pz', 'trk_pt', 
            'trk_inner_px', 'trk_inner_py', 'trk_inner_pz', 'trk_inner_pt', 
            'trk_outer_px', 'trk_outer_py', 'trk_outer_pz', 'trk_outer_pt',
            'trk_eta', 'trk_lambda', 'trk_cotTheta', 'trk_phi', 
            'trk_dxy', 'trk_dz', 'trk_dxyPV', 'trk_dzPV', 'trk_dxyClosestPV', 'trk_dzClosestPV',
            'trk_ptErr', 'trk_etaErr', 'trk_lambdaErr', 'trk_phiErr', 'trk_dxyErr', 'trk_dzErr',
            'trk_refpoint_x', 'trk_refpoint_y', 'trk_refpoint_z',
            'trk_nChi2', 'trk_nChi2_1Dmod', 'trk_ndof', 'trk_q', 
            'trk_nValid', 'trk_nLost', 'trk_nInactive', 'trk_nPixel', 'trk_nStrip', 
            'trk_nOuterLost', 'trk_nInnerLost', 'trk_nOuterInactive', 'trk_nInnerInactive',
            'trk_nPixelLay', 'trk_nStripLay', 'trk_n3DLay', 'trk_nLostLay', 'trk_nCluster'
        ]
        
        print(f"  Processing {len(batch_files)} files using {self.n_workers} workers...")
        
        worker_args = [(file_path, columns) for file_path in batch_files]
        
        data_dicts = []
        labels_list = []
        
        if self.n_workers > 1:
            with Pool(processes=self.n_workers) as pool:
                for i, result in enumerate(pool.imap(_process_single_file, worker_args), 1):
                    if result is not None:
                        features_dict, labels = result
                        data_dicts.append(features_dict)
                        labels_list.append(labels)
                    if i % 5 == 0 or i == len(batch_files):
                        print(f"    Processed {i}/{len(batch_files)} files...")
        else:
            for i, args in enumerate(worker_args, 1):
                result = _process_single_file(args)
                if result is not None:
                    features_dict, labels = result
                    data_dicts.append(features_dict)
                    labels_list.append(labels)
                if i % 5 == 0 or i == len(batch_files):
                    print(f"    Processed {i}/{len(batch_files)} files...")
        
        if not data_dicts:
            raise RuntimeError("No data could be loaded from any file in batch")
        
        print(f"  Successfully loaded {len(data_dicts)}/{len(batch_files)} files")
        print(f"  Concatenating {len(data_dicts)} file results...")
        
        # Concatenate features and labels
        concatenated_features = {}
        for col in columns:
            col_data = [d[col] for d in data_dicts]
            concatenated_features[col] = ak.concatenate(col_data, axis=0)
        
        concatenated_labels = ak.concatenate(labels_list, axis=0)
        
        print(f"  Converting to numpy arrays...")
        data_numpy = np.column_stack([
            ak.to_numpy(ak.flatten(concatenated_features[col])) 
            for col in columns
        ])
        labels_numpy = concatenated_labels.to_numpy()
        
        print(f"  Converting to PyTorch tensors...")
        batch_data = torch.from_numpy(data_numpy).float()
        batch_labels = torch.from_numpy(labels_numpy).float().unsqueeze(1)
        
        print(f"  Saving batch cache to {cache_file.name}...")
        try:
            torch.save({
                'data': batch_data,
                'labels': batch_labels
            }, cache_file)
            elapsed_time = time.time() - start_time
            print(f"  Batch processed in {elapsed_time:.2f}s - {len(batch_data):,} samples")
        except Exception as e:
            print(f"  Warning: Failed to save cache ({e})")
        
        return batch_data, batch_labels

    def __len__(self):
        if self.lazy_load:
            return self._total_samples
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.lazy_load:
            batch_idx, local_idx = self._find_batch_for_idx(idx)
            batch_data = self._load_batch_by_index(batch_idx)
            
            data = batch_data['data'][local_idx]
            label = batch_data['labels'][local_idx]
            
            sample = (data, label)
        else:
            sample = (self.data[idx], self.labels[idx])
        
        if self.transform:
            sample = self.transform(sample)
        return sample