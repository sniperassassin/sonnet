# Sonnet4Style-GPT — Quick Instructions

This file shows a minimal, copy-paste PowerShell workflow to create a virtual environment, install dependencies (including a CUDA-enabled PyTorch build), run a quick training job, and generate text using the project CLI.

Prerequisites
- Windows PowerShell
- Python 3.9+ available as `python3`/`python`
- NVIDIA drivers installed (verify with `nvidia-smi`)

1) Create & activate a virtual environment

```powershell
# from repo root (e.g. C:\github\sonnet\sonnet-gpt)
python3 -m venv .venv
# Activate in PowerShell
.\.venv\Scripts\Activate.ps1
```

After activation your prompt shows `(.venv)`.

2) Upgrade pip and install build tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

3) Install project requirements

```powershell
python -m pip install -r .\sonnet4style-gpt\sonnet4style-gpt\requirements.txt
```

4) Install CUDA-enabled PyTorch (if needed)

If you want GPU support make sure you install a CUDA-enabled torch wheel that matches your driver/runtime. Example for CUDA 12.1:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio --upgrade
```

If you don't need GPU or prefer the system package, skip this step.

5) (Optional) Install package in editable mode so `python -m sonnet_gpt.cli` is available

```powershell
cd .\sonnet4style-gpt\sonnet4style-gpt
python -m pip install -e .
cd ..\..
```

6) Verify that PyTorch and GPU are available

```powershell
python .\sonnet4style-gpt\sonnet4style-gpt\check_gpu.py
# Quick check
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

7) Run a short training job (tiny config)

```powershell
# From repo root (or from inside sonnet4style-gpt folder)
python -m sonnet_gpt.cli --config ./sonnet4style-gpt/sonnet4style-gpt/configs/tiny.json train


# If you installed editable package and are inside package folder:
cd .\sonnet4style-gpt\sonnet4style-gpt
python -m sonnet_gpt.cli --config .\configs\tiny.json train
```

- Checkpoints are saved to `checkpoints/model.pt` and `checkpoints/config.json` relative to the current working directory.

8) Generate text from a saved checkpoint

```powershell
python -m sonnet_gpt.cli --config ./configs/tiny.json generate --prompt "To be, or not to be" --ckpt_dir .\checkpoints
```

9) Deactivate venv and re-enter later

```powershell
deactivate
# To reactivate later
.\.venv\Scripts\Activate.ps1
```

Troubleshooting & tips
- If `torch.cuda.is_available()` is False inside the venv, ensure you installed the CUDA-enabled wheel into that venv (repeat step 4 while the venv is active).
- If you run out of GPU memory: use `configs/tiny.json`, reduce `batch_size`, or set `training.device` to `"cpu"` in the JSON.
- If you prefer conda, create a conda env and install `pytorch` + `pytorch-cuda` from the `pytorch` and `nvidia` channels instead of using pip.
- To run the CLI without editable install, set `PYTHONPATH` to the package directory:
  ```powershell
  $env:PYTHONPATH = ".\sonnet4style-gpt\sonnet4style-gpt"
  python -m sonnet_gpt.cli --config .\sonnet4style-gpt\sonnet4style-gpt\configs\tiny.json train
  ```

Where to add this file
- This `instructions.md` lives at `sonnet4style-gpt/sonnet4style-gpt/instructions.md` in the repository.

If you want, I can also: add these steps to the top-level `README.md`, create a short shell script for Windows PowerShell that automates venv creation + install, or run a quick smoke training run and paste the output here. Which of those would you like? 

Venv persistence (do I need to reinstall each time?)

- No — you do NOT need to reinstall packages every time you activate the same virtual environment. Packages installed into `.venv` persist until you delete that directory.
- Activation is per shell session; run `.\.venv\Scripts\Activate.ps1` each time you open a new shell to enter the environment.
- If you want to reproduce the environment on another machine or later, save a lock file:

```powershell
python -m pip freeze > requirements-locked.txt
```

Then recreate with:

```powershell
python -m pip install -r requirements-locked.txt
```

- Editable installs (`python -m pip install -e .`) also persist inside the venv — no re-install on activate.
- Reinstall only if you delete the venv, switch interpreters, or intentionally remove/upgrade packages.
