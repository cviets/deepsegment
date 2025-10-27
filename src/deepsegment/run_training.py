from .modules.get_data import MaskDataset
from .modules.training_module import train, validate
from .modules.training_helper import DiceCoefficient, salt_and_pepper_noise, launch_tensorboard
import torch.nn as nn
from dlmbl_unet import UNet
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import WeightedRandomSampler

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
        weighted=False
    )
    num_classes = 3
    
    val_size = int(val_ratio*len(augmented_data))
    train_size = len(augmented_data) - val_size

    train_set, val_set = random_split(augmented_data, [train_size, val_size])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # do not apply softmax activation b/c this is automatically encoded in CrossEntropyLoss
    # Since targets are trinary class indices (0=background,1=nucleus,2=cytoplasm), set out_channels=3
    unet = UNet(depth=unet_depth, in_channels=2, out_channels=num_classes, num_fmaps=num_fmaps, final_activation=None).to(device)

    class_weights = torch.tensor([0.55, 0.6, 0.49], dtype=torch.float32)
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.clamp(0.1, 10.0)
    # move weights to device when creating the loss below
    class_weights = class_weights.to(device)

    loss = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)
    # learning rate scheduler: reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # --- compute per-image (original dataset) image_weights ---
    # Suppose augmented_data.masks[i] is either a class-index map (H,W)
    # or a one-hot tensor (2,H,W) as in your get_data implementation.

    image_weights = []
    for i in range(len(augmented_data)):
        stored = augmented_data.masks[i]
        if not isinstance(stored, torch.Tensor):
            stored = torch.tensor(stored)

        # decode to class-index map
        if stored.dim() == 3 and stored.shape[0] == 2:
            class_map = torch.zeros(stored.shape[1:], dtype=torch.long)
            class_map[stored[1] == 1] = 1
            class_map[stored[0] == 1] = 2
        else:
            class_map = stored.long()

        # per-image class fractions
        binc = torch.bincount(class_map.flatten(), minlength=num_classes).float()
        frac = binc / binc.sum()

        # compute image weight from global class weights (example)
        # assume global class_weights computed earlier (length num_classes)
        img_w = float((class_weights.cpu() * frac).sum())  # weighted sum heuristic
        image_weights.append(img_w)

    # If train_set is a Subset, pick the weights for that subset in order:
    if hasattr(train_set, "indices"):
        weights_for_sampler = [image_weights[i] for i in train_set.indices]
    else:
        weights_for_sampler = image_weights

    # Normalize & avoid zeros
    weights_for_sampler = torch.tensor(weights_for_sampler, dtype=torch.double)
    weights_for_sampler = weights_for_sampler.clamp(min=1e-6)

    # Create sampler: replacement=True to allow oversampling
    sampler = WeightedRandomSampler(weights=weights_for_sampler, num_samples=len(weights_for_sampler), replacement=True)

    # DataLoader with sampler (do not set shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logger = SummaryWriter("runs/training")
    launch_tensorboard("runs")

    for epoch in range(epochs):
        train(unet, train_loader, optimizer, loss, epoch, device=device, log_image_interval=log_image_interval, log_interval=log_interval, tb_logger=logger)

        step = epoch * len(train_loader)
        val_loss, val_metric = validate(unet, val_loader, loss, metric=DiceCoefficient(), step=step, tb_logger=logger)

        # step scheduler with validation loss
        try:
            scheduler.step(val_loss)
        except Exception:
            # if val_loss is not numeric for some reason, skip scheduler step
            pass

        # log current LR
        current_lr = optimizer.param_groups[0]["lr"]
        logger.add_scalar(tag="learning_rate", scalar_value=current_lr, global_step=step)