import os
import cv2
import pickle
import random
import numpy as np
from scipy import sparse

from torch.utils.data import Dataset


class MultiOmniCityDataset(Dataset):
    def __init__(self, data_file, prompt, image_size=(1024, 512), source_image_size=(1024, 512), drop_context_ratio=0):
        self.data = []
        self.image_size = image_size
        self.source_image_size = source_image_size
        self.drop_context_ratio = drop_context_ratio
        with open(data_file, 'rt') as f:
            for line in f:
                segs = line.strip().split('\t')
                if len(segs) >= 5:
                    self.data.append({
                        'name': segs[0].strip(),
                        'target_hint': segs[1].strip(),
                        'target_image': segs[2].strip(),
                        'source_hint': segs[3].strip(),
                        'source_angle': round(float(segs[4].strip())) % 360,
                        'source_weight': segs[5].strip() if len(segs) >= 6 else '',
                        'prompt': prompt.strip(),
                    })
                else:
                    self.data.append({
                        'name': segs[0].strip(),
                        'target_hint': segs[1].strip(),
                        'target_image': segs[2].strip(),
                        'source_hint': segs[3].strip(),
                        'source_angle': 0,
                        'source_weight': '',
                        'prompt': prompt.strip(),
                    })


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        name = item['name']
        source_hint_filename = item['source_hint']
        target_hint_filename = item['target_hint']
        target_image_filename = item['target_image']
        source_angle = item['source_angle']
        source_weight_filename = item['source_weight']
        prompt = item['prompt']

        source_hint = cv2.imread(source_hint_filename)
        # Do not forget that OpenCV read images in BGR order.
        source_hint = cv2.cvtColor(source_hint, cv2.COLOR_BGR2RGB)
        source_hint = cv2.resize(source_hint, self.source_image_size)

        if source_weight_filename != '':
            source_weight = sparse.load_npz(source_weight_filename).toarray()
        else:
            pixel_num = np.prod(self.source_image_size) // (8*8)
            source_weight = np.zeros((pixel_num, pixel_num))

        target_hint = cv2.imread(target_hint_filename)
        # Do not forget that OpenCV read images in BGR order.
        target_hint = cv2.cvtColor(target_hint, cv2.COLOR_BGR2RGB)
        target_hint = cv2.resize(target_hint, self.image_size)

        target_image = cv2.imread(target_image_filename)
        # Do not forget that OpenCV read images in BGR order.
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        target_image = cv2.resize(target_image, self.image_size)

        # Normalize source images to [0, 1].
        source_hint = source_hint.astype(np.float32) / 255.0
        target_hint = target_hint.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target_image = (target_image.astype(np.float32) / 127.5) - 1.0

        # randomly drop source hint / context
        if self.drop_context_ratio > 0 and random.random() < self.drop_context_ratio:
            source_angle = 0
            source_hint = np.zeros_like(source_hint)
            source_weight = np.zeros_like(source_weight)

        return dict(jpg=target_image, txt=prompt,
                    source_hint=source_hint, source_angle=source_angle, source_weight=source_weight,
                    target_hint=target_hint, name=name)

