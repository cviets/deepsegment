import torch.nn as nn
import torch
import subprocess

# sorensen dice coefficient implemented in torch
# the coefficient takes values in two discrete arrays
# with values in {0, 1}, and produces a score in [0, 1]
# where 0 is the worst score, 1 is the best score
class DiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # the dice coefficient of two sets represented as vectors a, b ca be
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        intersection = (prediction * target).sum()
        union = (prediction * prediction).sum() + (target * target).sum()
        return 2 * intersection / union.clamp(min=self.eps)
    
def salt_and_pepper_noise(image, amount=0.05):
    """
    Add salt and pepper noise to an image
    """
    out = image.clone()
    num_salt = int(amount * image.numel() * 0.5)
    num_pepper = int(amount * image.numel() * 0.5)

    # Add Salt noise
    coords = [
        torch.randint(0, i - 1, [num_salt]) if i > 1 else [0] * num_salt
        for i in image.shape
    ]
    out[tuple(coords)] = 1

    # Add pepper noise
    coords = [
        torch.randint(0, i - 1, [num_pepper]) if i > 1 else [0] * num_pepper
        for i in image.shape
    ]
    out[tuple(coords)] = 0

    return out

def launch_tensorboard(log_dir):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        tensorboard_cmd = f"tensorboard --logdir={log_dir} --port={port}"
        process = subprocess.Popen(tensorboard_cmd, shell=True)
        print(
            f"TensorBoard started at http://localhost:{port}. \n"
            "If you are using VSCode remote session, forward the port using the PORTS tab next to TERMINAL."
        )
        return process