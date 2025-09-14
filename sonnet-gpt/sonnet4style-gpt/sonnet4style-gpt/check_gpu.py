import subprocess
import sys
import shutil

def run_nvidia_smi():
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found in PATH")
        return None
    try:
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
        return out
    except Exception as e:
        return f"nvidia-smi failed: {e}"

def check_torch():
    try:
        import torch
    except Exception as e:
        return {"installed": False, "error": str(e)}
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

def main():
    print("Checking torch and system GPU state")
    t = check_torch()
    if not t["installed"]:
        print("PyTorch not installed:", t.get("error"))
        sys.exit(1)
    print("PyTorch:", t["version"])
    print("CUDA available:", t["cuda_available"])
    print("CUDA runtime:", t["cuda_version"])
    print("Device:", t["device_name"])
    print()
    print("Running nvidia-smi (if available):")
    nv = run_nvidia_smi()
    if nv is None:
        print("nvidia-smi not found")
    else:
        print(nv)

    # exit code: 0 when CUDA available, 2 when torch installed but no CUDA, 1 when torch missing
    if t["cuda_available"]:
        sys.exit(0)
    else:
        sys.exit(2)

if __name__ == '__main__':
    main()
