from .modules.get_data import MaskDataset
from .modules.training_module import train, validate
from .modules.training_helper import DiceCoefficient, salt_and_pepper_noise
import torch.nn as nn
from dlmbl_unet import UNet
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

def run_training(root_dir, epochs, learning_rate, val_ratio, batch_size, num_workers, unet_depth, num_fmaps):

    augmented_data = MaskDataset(
        root_dir,
        v2.Compose(
            [v2.RandomRotation(45), v2.RandomCrop(256)]
        ),
        img_transform=v2.Compose([v2.Lambda(salt_and_pepper_noise)]),
    )
    
    val_size = int(val_ratio*len(augmented_data))
    train_size = len(augmented_data) - val_size

    train_set, val_set = random_split(augmented_data, [train_size, val_size])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    unet = UNet(depth=unet_depth, in_channels=1, out_channels=1, num_fmaps=num_fmaps).to(device)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logger = SummaryWriter("runs/training")


    for epoch in range(epochs):
        train(unet, train_loader, optimizer, loss, epoch, device=device)
        
        step = epoch*len(train_loader)
        validate(unet, val_loader, loss, metric=DiceCoefficient(), step=step, tb_logger=logger)