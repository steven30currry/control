import os
import cv2
import random
import pickle
import numpy as np

from torch.utils.data import Dataset


class OmniCityDataset(Dataset):
    def __init__(self, data_file, prompt, image_size=(1024, 512), drop_prompt_ratio=0):
        self.data = []
        self.image_size = image_size
        self.drop_prompt_ratio = drop_prompt_ratio
        with open(data_file, 'rt') as f:
            for line in f:
                segs = line.strip().split('\t')
                if len(segs) == 4:
                    self.data.append({
                        'name': segs[0].strip(),
                        'source': segs[1].strip(),
                        'target': segs[2].strip(),
                        'prompt': segs[3].strip(),
                    })
                else:
                    self.data.append({
                        'name': segs[0].strip(),
                        'source': segs[1].strip(),
                        'target': segs[2].strip(),
                        'prompt': prompt.strip(),
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        name = item['name']
        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # randomly drop text prompt
        if self.drop_prompt_ratio > 0 and random.random() < self.drop_prompt_ratio:
            prompt = ""

        if os.path.splitext(source_filename)[1] == '.pkl':
            source = pickle.load(open(source_filename, 'rb'))
        else:
            source = cv2.imread(source_filename)
            # Do not forget that OpenCV read images in BGR order.
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source = cv2.resize(source, self.image_size)

        target = cv2.imread(target_filename)
        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target, self.image_size)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, name=name)

