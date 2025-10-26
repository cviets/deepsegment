from .modules.get_data import MaskDataset
from .modules.training_module import train, validate
from .modules.training_helper import DiceCoefficient, salt_and_pepper_noise, launch_tensorboard
import torch.nn as nn
from dlmbl_unet import UNet
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

def run_training(root_dir, epochs, learning_rate, val_ratio, batch_size, crop_size, num_workers, log_image_interval, log_interval, unet_depth, num_fmaps):

    augmented_data = MaskDataset(
        root_dir,
        transform=v2.Compose([
            v2.RandomRotation(45), 
            v2.RandomVerticalFlip(0.5),
            v2.RandomAffine(10, shear=(-10,10,-10,10)),
            v2.RandomCrop(crop_size)   
        ]),
        img_transform=v2.Compose([
            v2.GaussianBlur(21, sigma=0.1),
            # v2.Lambda(salt_and_pepper_noise)
        ]),
        weighted=True
    )
    
    val_size = int(val_ratio*len(augmented_data))
    train_size = len(augmented_data) - val_size

    train_set, val_set = random_split(augmented_data, [train_size, val_size])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    unet = UNet(depth=unet_depth, in_channels=2, out_channels=2, num_fmaps=num_fmaps).to(device)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logger = SummaryWriter("runs/training")
    launch_tensorboard("runs")

    for epoch in range(epochs):
        train(unet, train_loader, optimizer, loss, epoch, device=device, log_image_interval=log_image_interval, log_interval=log_interval, tb_logger=logger)
        
        step = epoch*len(train_loader)
        validate(unet, val_loader, loss, metric=DiceCoefficient(), step=step, tb_logger=logger)