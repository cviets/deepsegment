import torch
import torch.nn.functional as F

def convert_to_onehot(mask, num_classes):
    mask_vis = torch.zeros(size=(mask.shape[0], num_classes, mask.shape[1], mask.shape[2]))
    assert mask_vis[:,0,:,:].shape == mask.shape, f"Shape mismatch {mask_vis[0].shape} vs {mask.shape}"
    mask_vis[:,0,:,:] = mask==2
    mask_vis[:,1,:,:] = mask==1

    return mask_vis

def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval,
    log_image_interval,
    tb_logger=None,
    device=None,
    early_stop=False,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (image, mask) in enumerate(loader):

        # if loss_weights is not None:
        #     loss_weights = loss_weights.to(device)
        image, mask = image.to(device), mask.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(image)
        loss = loss_function(prediction, mask)

        # If this is a CrossEntropy-style setup (logits + class-index mask), compute
        # per-class loss contributions for logging. Otherwise fall back to generic loss.

        per_pixel = F.cross_entropy(prediction, mask, reduction="none")

        # log per-class average loss to tensorboard if logger provided
        if tb_logger is not None:
            num_classes = prediction.shape[1]
            step = epoch * len(loader) + batch_id
            for c in range(num_classes):
                mask_c = (mask == c)
                if mask_c.any():
                    avg_c = per_pixel[mask_c].mean().item()
                else:
                    avg_c = float('nan')
                tb_logger.add_scalar(tag=f"train_loss_class_{c}", scalar_value=avg_c, global_step=step)
            

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id,
                    len(loader),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                image_ = torch.cat((image, torch.zeros(size=(image.shape[0], 1, image.shape[2], image.shape[3]))), axis=1)
                tb_logger.add_images(
                    tag="input", img_tensor=image_.to("cpu"), global_step=step
                )

                pred_idx = prediction.argmax(dim=1)

                mask_vis = convert_to_onehot(mask, num_classes=3)
                pred_vis = convert_to_onehot(pred_idx, num_classes=3)

                # pad to 3 channels if needed for tensorboard
                if mask_vis.shape[1] == 1:
                    mask_vis = torch.cat((mask_vis, torch.zeros(size=(mask_vis.shape[0], 1, mask_vis.shape[2], mask_vis.shape[3]))), axis=1)
                if pred_vis.shape[1] == 1:
                    pred_vis = torch.cat((pred_vis, torch.zeros(size=(pred_vis.shape[0], 1, pred_vis.shape[2], pred_vis.shape[3]))), axis=1)

                tb_logger.add_images(
                    tag="target", img_tensor=mask_vis.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=pred_vis.to("cpu").detach(),
                    global_step=step,
                )

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break

# run validation after training epoch
def validate(
    model,
    loader,
    loss_function,
    metric,
    step=None,
    tb_logger=None,
    device=None,
):
    """
    Evaluate model performance on validation data.

    Args:
        model: PyTorch model to validate
        loader: DataLoader containing validation data
        loss_function: Loss function to compute validation loss between prediction and target.
        metric: Metric function to evaluate segmentation against ground-truth labels.
        step: Current training step for logging (required if tb_logger provided)
        tb_logger: TensorBoard logger for recording metrics (optional)
        device: Torch device to run validation on (optional)

    Returns:
        None
    """

    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set model to eval mode
    model.eval()
    model.to(device)

    # running loss and metric values
    val_loss = 0
    val_metric = 0

    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for image, mask in loader:
            image, mask = image.to(device), mask.to(device)

            # We *usually* want the target to be the same type as the prediction
            # however this is very dependent on your choice of loss function and
            # metric. If you get errors such as "RuntimeError: Found dtype Float but expected Short"
            # then this is where you should look.
            prediction = model(image)

            # For CrossEntropy-style predictions (logits + class-index mask) compute per-class
            # validation loss contributions for logging and metric computation. Otherwise fall back
            # to the generic loss/metric code path.

            per_pixel = F.cross_entropy(prediction, mask, reduction="none")
            batch_loss = loss_function(prediction, mask)
            # metric: try to compute per-class dice by converting to one-hot if metric expects that
            try:
                pred_idx = prediction.argmax(dim=1)  # (N,H,W)
                pred_onehot = F.one_hot(pred_idx, num_classes=prediction.shape[1]).permute(0,3,1,2).float()
                target_onehot = F.one_hot(mask, num_classes=prediction.shape[1]).permute(0,3,1,2).float()
                metric_val = metric(pred_onehot, target_onehot).item()
            except Exception:
                # fallback: pass class indices
                metric_val = metric(pred_idx, mask).item()

            # log per-class validation losses when a tb_logger is provided
            if tb_logger is not None:
                num_classes = prediction.shape[1]
                for c in range(num_classes):
                    mask_c = (mask == c)
                    if mask_c.any():
                        avg_c = per_pixel[mask_c].mean().item()
                    else:
                        avg_c = float('nan')
                    tb_logger.add_scalar(tag=f"val_loss_class_{c}", scalar_value=avg_c, global_step=step if step is not None else 0)

            val_loss += batch_loss
            val_metric += metric_val

    # normalize loss and metric
    val_loss /= len(loader)
    val_metric /= len(loader)

    if tb_logger is not None:
        assert (
            step is not None
        ), "Need to know the current step to log validation results"
        tb_logger.add_scalar(tag="val_loss", scalar_value=val_loss, global_step=step)
        tb_logger.add_scalar(
            tag="val_metric", scalar_value=val_metric, global_step=step
        )
        # we always log the last validation images
        image_ = torch.cat((image, torch.zeros(size=(image.shape[0], 1, image.shape[2], image.shape[3]))), axis=1)
        tb_logger.add_images(tag="val_input", img_tensor=image_.to("cpu"), global_step=step)

        pred_idx = prediction.argmax(dim=1)

        mask_vis = convert_to_onehot(mask, num_classes=3)
        pred_vis = convert_to_onehot(pred_idx, num_classes=3)

        # pad to 3rd channel for tensorboard if needed
        if pred_vis.shape[1] == 1:
            pred_vis = torch.cat((pred_vis, torch.zeros(size=(pred_vis.shape[0], 1, pred_vis.shape[2], pred_vis.shape[3]))), axis=1)
        if mask_vis.shape[1] == 1:
            mask_vis = torch.cat((mask_vis, torch.zeros(size=(mask_vis.shape[0], 1, mask_vis.shape[2], mask_vis.shape[3]))), axis=1)

        tb_logger.add_images(tag="val_target", img_tensor=mask_vis.to("cpu"), global_step=step)
        tb_logger.add_images(tag="val_prediction", img_tensor=pred_vis.to("cpu"), global_step=step)

    print(
        "\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n".format(
            val_loss, val_metric
        )
    )

    # return values so callers (e.g. run_training) can step schedulers based on val loss
    return val_loss, val_metric