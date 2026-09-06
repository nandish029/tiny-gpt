import torch

def get_device() -> torch.device:
    """Returns the appropriate PyTorch device based on CUDA availability."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_device_info() -> dict:
    """Returns a dictionary containing basic information about the available device."""
    info = {
        "cuda_available": torch.cuda.is_available(),
    }
    
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_type"] = "cuda"
    else:
        info["device_name"] = "CPU"
        info["device_type"] = "cpu"
        
    return info
