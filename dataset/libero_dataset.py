from typing import List, Dict, Tuple
import os, glob
from pathlib import Path
import h5py
import json
from PIL import Image, ImageOps
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from libero.libero import benchmark, get_libero_path

from utils.common import set_seed


class LiberoDataset(Dataset):
    """
    Single-view frame dataset for Libero demos.
    
    Return: (image, meta)
    - image: float tensor in [0, 1]. shape: (3, H, W)
    - meta
    """
    def __init__(self,
                 dataset_path: str,
                 view: str,
                 resize: Tuple[int,int]=(128, 128),
                 frame_stride: int=3,
                 seed: int=42,
                 ):
        super().__init__()
        set_seed(seed)
        
        if dataset_path is None:
            dataset_path = get_libero_path('dataset')
        
        self.dataset_path = dataset_path
        self.view = view
        self.resize = resize
        self.frame_stride = frame_stride

        self._preload_data()
        
    def _preload_data(self):
        dataset_files = [str(path) for path in Path(self.dataset_path).rglob('*.hdf5')]
        self.index = []
        for file in dataset_files:
            f = h5py.File(file, 'r')
            data = f['data']
            problem_info = json.loads(data.attrs['problem_info'])
            instruction = problem_info['language_instruction']
            demo_keys = list(data.keys())
            for key in demo_keys:
                demo = data[key]
                image_obs = demo['obs'][self.view][()]
                T = image_obs.shape[0]
                for t in range(0, T, self.frame_stride):
                    self.index.append((file, key, t))
            f.close()
        f.close()
        
            
    def __len__(self):
        return len(self.index)
    
    def _read_image(self, file, demo_key, t) -> np.ndarray:
        data = file['data']
        demo = data[demo_key]
        img = demo['obs'][self.view][t]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def __getitem__(self, index):
        file_path, demo_key, t = self.index[index]
        file = h5py.File(file_path, 'r')
        img = self._read_image(file, demo_key, t)
        img = Image.fromarray(img)
        img = ImageOps.flip(img)
        img = img.resize(self.resize, Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        # img = img.transpose(2, 0, 1)
        return torch.from_numpy(img), {'file': file_path, 'demo': demo_key, 't': t}
    

class LiberoImageDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 shape_meta: dict,
                 seq_len: int=8,
                 frame_stride: int=1,
                 mode: str='train',
                 val_ratio: float=0.05,
                 seed: int=42,
                 ):
        super().__init__()
        set_seed(seed)
        
        assert mode in ['train', 'val'], f'Mode should be "train" or "val", not {mode}.'
        self.mode = mode
        
        if dataset_path is None:
            dataset_path = get_libero_path('dataset')
        
        self.dataset_path = dataset_path
        self.shape_meta = shape_meta
        
        self.views = []
        self.low_dims = []
        self.image_size = None
        for k, v in shape_meta['obs'].items():
            t = v.get('type', 'low_dim')
            if t == 'rgb':
                self.views.append(k)
                self.image_size = v['shape']
            else:
                self.low_dims.append(k)
        
        self.num_views = len(self.views)
        self.seq_len = seq_len
        self.frame_stride = frame_stride
        self.val_ratio = val_ratio
        
        self._build_index()
    
    def _build_index(self):
        dataset_files = [str(path) for path in Path(self.dataset_path).rglob('*.hdf5')]
        random.shuffle(dataset_files)
        trajectories = []
        for file in dataset_files:
            try:
                with h5py.File(file, 'r') as f:
                    data = f['data']
                    for key in data.keys():
                        trajectories.append((file, key))
                    f.close()
            except Exception:
                continue
        
        random.shuffle(trajectories)
        n_val = int(len(trajectories) * self.val_ratio)
        n_train = int(len(trajectories) - n_val)
        chosen = trajectories[:n_train] if self.mode == 'train' else trajectories[n_train:]
        self.index = []
        for (file, key) in chosen:
            with h5py.File(file, 'r') as f:
                imgs0 = f['data'][key]['obs'][self.views[0]]
                T = imgs0.shape[0]
                max_start = T - (self.seq_len-1) * self.frame_stride - 1
                if max_start < 0:
                    continue
                for s in range(0, max_start+1):
                    self.index.append((file, key, s))
    
    def _read_image(self, img) -> np.ndarray:
        """ Return image (C, H, W) in (0., 1.) """
        if img.ndim == 3 and img.shape[0] in (1, 3):
            # (C, H, W) -> (H, W, C)
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        img = ImageOps.flip(img)
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))      # (C, H, W)
        return img
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file, key, s = self.index[idx]
        item = {}
        with h5py.File(file, 'r') as f:
            data = f['data']
            problem_info = json.loads(f['data'].attrs['problem_info'])
            instruction = problem_info.get('language_instruction', '')
            
            demo = data[key]
            ts = [s + i*self.frame_stride for i in range(self.seq_len)]
            for v in self.views:
                img = demo['obs'][v]
                frames = [self._read_image(img[t]) for t in ts]
                imgs = np.stack(frames, axis=0)     # (T, C, H, W)
                item[v] = torch.from_numpy(imgs)
            item['instruction'] = instruction
            f.close()
        item['meta'] = {'file': file, 'demo': key, 'start': s}
        return item
    
        
# if __name__=='__main__':
#     from torchvision.utils import save_image
#     shape_meta = {
#         'obs': {'agentview_rgb': {'shape': [128, 128, 3], 'type': 'rgb'},
#                 'eye_in_hand_rgb': {'shape': [128, 128, 3], 'type': 'rgb'},},
#         'action': {'shape': [7]}
#     }
#     dataset = LiberoImageDataset(dataset_path='data/libero/libero_10', shape_meta=shape_meta)
#     data = dataset[0]
#     img = data['agentview_rgb']
#     save_image(img, 'test.png')