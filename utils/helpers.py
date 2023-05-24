import os
import random

import numpy as np
import torch


def set_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def create_result_directory(path, args):
    dir = f"{args.dataset}-{args.arch}-"
    return os.path.join(
        path,
        f"{dir}rho{args.rho}-seed{args.seed}-bs{args.batch_size}-lr{args.lr}-wd{args.wd}",
    )


def find_latest_checkpoint(paths):
    max_value, checkpoint = -1, ""
    for path in paths:
        epoch = int(os.path.split(path)[-1].split("-")[1].replace(".pt", ""))
        if max_value < epoch:
            max_value = epoch
            checkpoint = path
    return checkpoint


def save_model(model, directory, filename="model.pt"):
    device = next(model.parameters()).device
    model.cpu()

    path_to_model = os.path.join(directory, filename)
    torch.save(model.state_dict(), path_to_model)

    model.to(device)  # restore device
