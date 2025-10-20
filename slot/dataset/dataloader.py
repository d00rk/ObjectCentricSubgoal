from typing import Tuple
import torch
from torch.utils.data import DataLoader
from slot.dataset.libero_dataset import LiberoImageDataset, LiberoSubgoalImageDataset


def get_libero_image_dataloader(dataset_path: str,
                                view: str,
                                resize: Tuple[int, int]=(128, 128),
                                val_ratio: float=0.1,
                                seed: int=42,
                                train_batch_size: int=32,
                                train_num_workers: int=0,
                                train_pin_memory: bool=True,
                                train_persistent_workers: bool=False,
                                val_batch_size: int=32,
                                val_num_workers: int=0,
                                val_pin_memory: bool=True,
                                val_persistent_workers: bool=False) -> DataLoader:
    train_dataset = LiberoImageDataset(dataset_path=dataset_path,
                                       view=view,
                                       resize=resize,
                                       mode='train',
                                       val_ratio=val_ratio,
                                       seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=train_num_workers, pin_memory=train_pin_memory, persistent_workers=train_persistent_workers)
    
    val_dataset = LiberoImageDataset(dataset_path=dataset_path,
                                     view=view,
                                     resize=resize,
                                     mode='val',
                                     val_ratio=val_ratio,
                                     seed=seed)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers, pin_memory=val_pin_memory, persistent_workers=val_persistent_workers)
    
    return train_loader, val_loader


def get_libero_subgoal_image_dataloader(dataset_path: str,
                                        view: str,
                                        resize: Tuple[int, int]=(128, 128),
                                        val_ratio: float=0.1,
                                        seed: int=42,
                                        train_batch_size: int=32,
                                        train_num_workers: int=0,
                                        train_pin_memory: bool=True,
                                        train_persistent_workers: bool=False,
                                        val_batch_size: int=32,
                                        val_num_workers: int=0,
                                        val_pin_memory: bool=True,
                                        val_persistent_workers: bool=False) -> DataLoader:
    train_dataset = LiberoSubgoalImageDataset(dataset_path=dataset_path,
                                       view=view,
                                       resize=resize,
                                       mode='train',
                                       val_ratio=val_ratio,
                                       seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=train_num_workers, pin_memory=train_pin_memory, persistent_workers=train_persistent_workers)
    
    val_dataset = LiberoSubgoalImageDataset(dataset_path=dataset_path,
                                     view=view,
                                     resize=resize,
                                     mode='val',
                                     val_ratio=val_ratio,
                                     seed=seed)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers, pin_memory=val_pin_memory, persistent_workers=val_persistent_workers)
    
    return train_loader, val_loader