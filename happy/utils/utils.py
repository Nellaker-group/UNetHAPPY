import time
from collections import OrderedDict
from pathlib import Path

import GPUtil
import torch


def set_gpu_device():
    print(GPUtil.showUtilization())
    device_ids = GPUtil.getAvailable(
        order="memory", limit=1, maxLoad=0.3, maxMemory=0.3
    )
    while not device_ids:
        print("No GPU avail.")
        time.sleep(10)
        device_ids = GPUtil.getAvailable(
            order="memory", limit=1, maxLoad=0.3, maxMemory=0.3
        )
    device_id = str(device_ids[0])
    print(f"Using GPU number {device_id}")
    return device_id


def get_device():
    if torch.cuda.is_available():
        device_id = set_gpu_device()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        device = f"cuda:{device_id}"
    else:
        device = "cpu"
    return device


def get_project_dir(project_name):
    root_dir = Path(__file__).absolute().parent.parent.parent
    return root_dir / "projects" / project_name


def load_weights(state_dict, model):
    # Removes the module string from the keys if it's there.
    if "module.conv1.weight" in state_dict.keys():
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    return model
