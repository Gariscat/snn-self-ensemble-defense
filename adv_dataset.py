import os
import numpy as np
from typing import Callable, Optional
from spikingjelly.datasets import DVS128Gesture

class DVS128GestureWithAdv(DVS128Gesture):
    def __init__(
            self,
            root: str,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            adversarial_data_root: Optional[str] = None,  # New parameter for adversarial data
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root, 
            train, 
            data_type, 
            frames_number, 
            split_by, 
            duration, 
            custom_integrate_function, 
            custom_integrated_frames_dir_name, 
            transform, 
            target_transform
        )
        self.adversarial_data_root = adversarial_data_root

    def load_adversarial_data(self) -> dict:
        if self.adversarial_data_root is None:
            return {}
        
        adversarial_data = {}
        for label in range(11):  # 11 gesture classes
            label_dir = os.path.join(self.adversarial_data_root, 'train', str(label))
            if os.path.exists(label_dir):
                adversarial_data[label] = [
                    os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.npz')
                ]
        return adversarial_data

    def __len__(self):
        original_length = super().__len__()
        adversarial_data = self.load_adversarial_data()
        adversarial_length = sum(len(files) for files in adversarial_data.values())
        return original_length + adversarial_length

    def __getitem__(self, idx):
        original_data = super().__getitem__(idx)
        
        adversarial_data = self.load_adversarial_data()
        adversarial_samples = []
        for label, files in adversarial_data.items():
            for file in files:
                adversarial_sample = np.load(file)
                adversarial_samples.append({
                    't': adversarial_sample['t'],
                    'x': adversarial_sample['x'],
                    'y': adversarial_sample['y'],
                    'p': adversarial_sample['p'],
                    'label': label
                })
        
        if idx < len(original_data):
            return original_data
        else:
            return adversarial_samples[idx - len(original_data)]
