import numpy as np
import torchvision
from torch.utils.data import Dataset
import cv2

# ovo je template razreda s predmeta Obrada Informacija na FER-u (lv5 - ak. god. 2023/24)
class MNISTLocalizationDataset(Dataset):

    def __init__(self, image_size=64, transform=None, train_set=False, min_size = 14, max_size = 56):
        self.image_size = image_size
        self.transform = transform
        
        self.min_size = min_size
        self.max_size = max_size

        self.set = torchvision.datasets.MNIST('./data/', train=train_set, download=True)
        self.position_cache = [-1] * len(self.set)

    def __len__(self):
        return len(self.set)

    def __getitem__(self, idx):        
        if self.position_cache[idx] == -1:
            # generate random size between [min_size, max_size]
            new_size = int(np.random.uniform(self.min_size, self.max_size + 1))

            # generating random positions
            x_pos = int(np.random.uniform(3, self.image_size - new_size - 2))
            y_pos = int(np.random.uniform(3, self.image_size - new_size - 2))
            self.position_cache[idx] = (x_pos, y_pos, new_size)
        
        x_pos, y_pos, new_size = self.position_cache[idx]
        
        # resize original image
        original = np.array(self.set[idx][0])
        if new_size > 28:
            flag = cv2.INTER_CUBIC
                
        else:
            flag = cv2.INTER_AREA
        resized = cv2.resize(original, (new_size, new_size), interpolation=flag)

        canvas = np.zeros((self.image_size, self.image_size, 1), dtype=np.uint8)
        canvas[y_pos:(y_pos+new_size), x_pos:(x_pos+new_size), 0] = resized

        x_pos = float(x_pos)
        y_pos = float(y_pos)

        if self.transform is not None:
            canvas = self.transform(canvas)

        return canvas, self.set[idx][1], (x_pos, y_pos, x_pos+new_size, y_pos+new_size)