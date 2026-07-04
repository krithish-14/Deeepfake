Deepfake Detection

This repository contains:

- `api/` — Django backend serving the verification API and frontend.
- `frontend/` — Single-page UI (served by Django in this project).
- `ml/` — Training, evaluation, and model code (ConvNeXt detector, dataset loader).

Quick start

1. Create and activate a Python virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ".\.venv\Scripts\Activate.ps1"
```

2. Install dependencies:

```powershell
pip install -r api/requirements.txt
pip install -r ml/requirements.txt
```

If you are on Windows and see NumPy warnings about MINGW-W64, install a stable NumPy release with:

```powershell
.\.venv\Scripts\python.exe -m pip install "numpy<1.26"
```

3. Run Django server (from `api`):

```powershell
cd api
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

4. Train a model (from `ml`):

```powershell
cd ml
python create_minimal_dataset.py    # optional: create small dataset for smoke test
python train_real.py --data-dir .\datasets\ffpp --epochs 10 --batch-size 16
```

Notes

- Do not commit large model weight files (`ml/weights/`) — they're ignored by `.gitignore`.
- To share trained weights use GitHub Releases, cloud storage, or Git LFS.

If you want, I can prepare the git commit and push instructions or help configure SSH keys for pushing to GitHub.
