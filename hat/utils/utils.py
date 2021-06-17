import time
import os
from collections import OrderedDict

import GPUtil


def print_gpu_stats():
    print(GPUtil.showUtilization())
    deviceIDs = GPUtil.getAvailable(order="memory", limit=1, maxLoad=0.3, maxMemory=0.3)
    while not deviceIDs:
        print("No GPU avail.")
        time.sleep(10)
        deviceIDs = GPUtil.getAvailable(
            order="memory", limit=1, maxLoad=0.3, maxMemory=0.3
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceIDs[0])
    print("Using GPU number ", os.environ["CUDA_VISIBLE_DEVICES"])
    return os.environ["CUDA_VISIBLE_DEVICES"]


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
