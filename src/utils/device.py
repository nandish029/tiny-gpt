import torch

def resolve_device(device_str: str = "auto") -> torch.device:
    """
    Resolves the execution device.

    Args:
        device_str: One of "auto", "cpu", "cuda"

    Returns:
        torch.device
    """
    device_str = device_str.lower()

    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cpu":
        return torch.device("cpu")
    elif device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Device set to 'cuda', but CUDA is not available on this machine.")
        return torch.device("cuda")
    else:
        raise ValueError(f"Invalid device specified: '{device_str}'. Expected 'auto', 'cpu', or 'cuda'.")

def get_device() -> torch.device:
    """Legacy helper. Defaults to auto."""
    return resolve_device("auto")

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
