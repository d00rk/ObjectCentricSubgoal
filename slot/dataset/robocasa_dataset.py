from typing import Union, List, Dict, Tuple
import os, glob
from pathlib import Path
import h5py
import json
from PIL import Image, ImageOps
import tqdm
import copy
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from slot.utils.common import set_seed


class RobocasaImageDataset(Dataset):
    """
    Single-view frame dataset for Robocasa demos.
    
    Return: {current_image, file, demo, current_step, instruction}
    - current_image: (C, H, W)
    """
    def __init__(self,
                 dataset_path: Union[str, List[str]],
                 view: str='robot0_agentview_left_image',
                 val_ratio: float=0.1,
                 seed: int=42,
                 ):
        super().__init__()
        set_seed(seed)
        
        if dataset_path is None:
            raise ValueError("Please provide 'dataset_path'.")
        
        self.dataset_path = dataset_path
        self.view = view
        self.val_ratio = val_ratio

        self._preload_data()
        
    def _preload_data(self):
        if isinstance(self.dataset_path, List):
            dataset_files = [str(path) for dpath in self.dataset_path for path in Path(dpath).rglob('*.hdf5')]
        else:
            dataset_files = [str(path) for path in Path(self.dataset_path).rglob('*.hdf5')]
        index = []
        for file in tqdm.tqdm(dataset_files, desc=f'Preloading dataset', leave=False):
            f = h5py.File(file, 'r')
            data = f['data']
            demo_keys = list(data.keys())
            for key in demo_keys:
                demo = data[key]
                ep_meta = json.loads(demo.attrs['ep_meta'])
                instruction = ep_meta['lang']
                image_obs = demo['obs'][self.view][()]
                T = image_obs.shape[0]
                for t in range(0, T):
                    index.append((file, key, t, instruction))
            f.close()
        
        random.shuffle(index)
        n_val = int(len(index) * self.val_ratio)
        n_train = int(len(index) - n_val)
        assert n_train + n_val == len(index)
        train_chosen = index[:n_train]
        val_chosen = index[n_train:]
        self.index = train_chosen
        self._val_index = val_chosen
    
    def get_val_dataset(self):
        dataset = copy.deepcopy(self)
        dataset.index = self._val_index
        return dataset
        
    def __len__(self):
        return len(self.index)
    
    def _read_image(self, file, demo_key, t) -> np.ndarray:
        data = file['data']
        demo = data[demo_key]
        img = demo['obs'][self.view][t]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path, demo_key, t, instruction = self.index[idx]
        file = h5py.File(file_path, 'r')
        img = self._read_image(file, demo_key, t)               # (H, W, C)
        img = Image.fromarray(img)
        img = ImageOps.flip(img)
        img = np.array(img)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1).contiguous()                 # (C, H, W)
        file.close()
        
        item = {
            'current_image': img,
            'file': file_path,
            'demo': demo_key,
            'current_step': t,
            'instruction': instruction
        }
        
        return item


class LiberoSubgoalImageDataset(Dataset):
    """
    Single-view frame dataset for Robocasa demos.
    
    Return: {current_image, subgoal_image, file, demo, current_step, subgoal_step, instruction}
    - current image: image of step t, float tensor in [0, 1]. shape: (C, H, W)
    - subgoal image: image of subgoal at step t, float tensor in [0, 1]. shape: (C, H, W)
    """
    def __init__(self,
                 dataset_path: str,
                 view: str='robot0_agentview_left_image',
                 resize: Tuple[int, int]=(128, 128),
                 val_ratio: float=0.05,
                 seed: int=42,
                 ):
        super().__init__()
        set_seed(seed)
        
        if dataset_path is None:
            raise ValueError("Please provide 'dataset_path'.")
        
        self.dataset_path = dataset_path
        self.view = view
        self.resize = resize
        self.val_ratio = val_ratio
        
        self._preload_data()
    
    def _preload_data(self):
        if isinstance(self.dataset_path, List):
            dataset_files = [str(path) for dpath in self.dataset_path for path in Path(dpath).rglob('*.hdf5')]
        else:
            dataset_files = [str(path) for path in Path(self.dataset_path).rglob('*.hdf5')]
        index = []
        for file in tqdm.tqdm(dataset_files, desc=f'Preloading dataset', leave=False):
            try:
                with h5py.File(file, 'r') as f:
                    data = f['data']
                    demo_keys = list(data.keys())
                    for key in demo_keys:
                        demo = data[key]
                        ep_meta = json.loads(demo.attrs['ep_meta'])
                        instruction = ep_meta['lang']
                        subgoal_indices = demo['uvd_subgoal_indices'][()]
                        image_obs = demo['obs'][self.view][()]
                        T = image_obs.shape[0]
                        i = 0
                        for t in range(0, T):
                            subgoal_index = subgoal_indices[i]
                            if t >= subgoal_index:
                                i += 1
                                subgoal_index = subgoal_indices[i]
                            index.append((file, key, t, subgoal_index, instruction))
                    f.close()
            except Exception:
                continue
        
        random.shuffle(index)
        n_val = int(len(index) * self.val_ratio)
        n_train = int(len(index) - n_val)
        assert n_train + n_val == len(index)
        
        train_index = index[:n_train]
        val_index = index[n_train:]
        self.index = train_index
        self._val_index = val_index
    
    def get_val_dataset(self):
        dataset = copy.deepcopy(self)
        dataset.index = self._val_index
        return dataset
    
    def __len__(self):
        return len(self.index)
    
    def _read_image(self, file, demo_key, t) -> np.ndarray:
        data = file['data']
        demo = data[demo_key]
        img = demo['obs'][self.view][t]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path, demo_key, current_t, subgoal_t, instruction = self.index[idx]
        file = h5py.File(file_path, 'r')
        
        current_img = self._read_image(file, demo_key, current_t)
        current_img = Image.fromarray(current_img)
        current_img = ImageOps.flip(current_img)
        current_img = np.array(current_img)
        current_img = torch.from_numpy(current_img)
        current_img = current_img.permute(2, 0, 1).contiguous()           # (C, H, W)
        
        subgoal_img = self._read_image(file, demo_key, subgoal_t)
        subgoal_img = Image.fromarray(subgoal_img)
        subgoal_img = ImageOps.flip(subgoal_img)
        subgoal_img = np.array(subgoal_img)
        subgoal_img = torch.from_numpy(subgoal_img)
        subgoal_img = subgoal_img.permute(2, 0, 1).contiguous()           # (C, H, W)
        
        file.close()
        
        item = {
            'current_image': current_img,
            'subgoal_image': subgoal_img,
            'file': file_path,
            'demo': demo_key,
            'current_step': current_t,
            'subgoal_step': subgoal_t,
            'instruction': instruction
        }
        
        return item