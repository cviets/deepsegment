import torch
from torch.utils.data import Dataset
import os
from torchvision.transforms import v2
import tifffile
import numpy as np

def minmax(inp):

    if inp.ndim > 2:
        return np.array([minmax(elt) for elt in inp])
    
    min_new = -1
    max_new = 1

    original_min = np.min(inp)
    original_max = np.max(inp)
    original_range = original_max - original_min

    new_range = max_new - min_new

    if original_range == 0:
        return inp
    return ((inp - original_min) / original_range) * new_range + min_new

class MaskDataset(Dataset):
    """
    A PyTorch dataset to load images and their corresponding masks
    """
    def __init__(self, root_dir, transform=None, img_transform=None, weighted=True):
        
        self.root_dir = os.path.expanduser(root_dir)
        self.transform = (transform)
        self.img_transform = img_transform
        self.weighted = weighted

        image_dir = os.path.join(root_dir, "images")
        mask_dir = os.path.join(root_dir, "masks")

        # TODO: reorder masks to be in same order as images
        
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
        assert len(self.images) > 0, "No training images found"
        assert len(self.images) == len(self.masks), f"Must have equal amounts of training images ({len(self.images)}) as annotated masks ({len(self.masks)})"

        # initial_img_transforms = v2.Compose([
        #     v2.Normalize([0]*2, [1]*2)
        # ])

        for i in range(len(self.images)):
            image = tifffile.imread(os.path.join(image_dir, self.images[i]))
            assert image.ndim == 3, "Image must be format CYX"
            image = np.stack((image[0], image[2]), axis=0)
            assert image.ndim == 3, image.ndim
            assert image.shape[0] == 2

            mask_ = np.load(os.path.join(mask_dir, self.masks[i]))["arr_0"][2]
            assert np.all(np.logical_or(np.logical_or(mask_==0, mask_==1), mask_==2)),\
                f"All mask values must be 0 (background), 1 (nucleus), or 2 (cytoplasm). Failed on {self.masks[i]}"
            
            mask = np.zeros(shape=(2, mask_.shape[0], mask_.shape[1]))
            mask[1] = mask_==1
            mask[0] = mask_==2

            image = minmax(image)
            image = torch.tensor(image, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)
            # image = initial_img_transforms(image)
            self.images[i] = image
            self.masks[i] = mask
            # self.masks[i] = mask.unsqueeze(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform is not None:
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)

            seed = torch.seed()
            torch.manual_seed(seed)
            mask = self.transform(mask)

        if self.img_transform is not None:
            image = self.img_transform(image)

        weights = None
        # play with weights if get bad model
        if self.weighted:
            # make sure model is not punished regardless of its guess for true background
            weights = torch.zeros_like(mask)
            weights[mask != 0] = 1

            # from MBL:
            # weights[mask == 0] = np.clip(
            #     mask.numel()
            #     / 2
            #     / (mask.numel() - mask.sum()),
            #     0.1,
            #     10.0,
            # )
            # weights[mask != 0] = np.clip(
            #     mask.numel() / 2 / mask.sum(), 0.1, 10.0
            # )

        return image, mask, weights