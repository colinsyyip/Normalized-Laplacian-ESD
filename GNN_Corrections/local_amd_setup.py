from sdnext import installer
import os


def local_setup(device_num: int = 0):
    """
    Runs local setup for ZLUDA using SDNEXT. 

    args:
        - device_num [int]: OS device number for GPU. Default is 0 for single GPU machines (non-integrated)
    """
    installer.ensure_base_requirements() 
    installer.check_version()
    installer.check_torch()

    import torch

    print("OS Env. Check: %s" % os.getenv("PYTORCH_NVML_BASED_CUDA_CHECK"))
    print("Torch C CUDA Device Count: %s" % torch._C._cuda_getDeviceCount())

    if torch.cuda.is_available():
        print("torch running on CUDA [ZLUDA]: %s" % torch.cuda.get_device_name(device_num))
        return True
    else:
        raise ValueError("torch cannot find CUDA [ZLUDA] enabled device")
    

if __name__ == "__main__":
    local_setup()