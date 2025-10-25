from src.deepsegment.modules.get_data import MaskDataset
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms.v2 as v2

def main(root_dir, save_dir, save_normed_dir):
    data = MaskDataset(root_dir, weighted=True)
    loader = DataLoader(data, batch_size=1, shuffle=True)

    for image, mask, weights in loader:
        np.save(save_dir, image)
        initial_img_transforms = v2.Compose([
            v2.Normalize([0]*2, [1]*2, inplace=True)
        ])
        initial_img_transforms(image)
        np.save(save_normed_dir, image)
        break