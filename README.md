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

Deployment (Frontend on Vercel, Backend on Render)

1. Frontend (Vercel)

- Ensure `frontend/config.json` contains your backend URL, for example:

```
{
	"API_BASE_URL": "https://your-backend.example.com"
}
```

- Push the repo to GitHub and connect the project to Vercel. `vercel.json` is already configured to serve `frontend/index.html`.

2. Backend (Render using Docker)

- The backend Dockerfile is `api/Dockerfile`. Render can build directly from the repo (select Docker). Configure the following environment variables in Render's dashboard or as secrets:

	- `DATABASE_URL` (Postgres or other DB)
	- `DJANGO_SECRET_KEY`
	- `DJANGO_DEBUG` (=False for production)
	- `VERCEL_BLOB_ENDPOINT`, `VERCEL_BLOB_BUCKET`, `VERCEL_BLOB_BASE_URL` (optional, for media storage)
	- `REDIS_URL` (optional)

- Local smoke-test (build and run):

```bash
cd api
docker build -t deepfake-backend .
docker run -e DJANGO_SECRET_KEY=dev -p 8000:8000 deepfake-backend
```

3. CORS / connectivity

- The backend already sets permissive CORS headers via `api/api/settings.py` middleware. When using production origins, update `ALLOWED_HOSTS` and prefer a stricter CORS policy.

4. Post-deploy

- Deploy frontend to Vercel, then deploy backend to Render and ensure `frontend/config.json` points at the backend URL. Run the app and check `/verify/` endpoints.

If you want, I can:

- Create a `render.yaml` for Render service configuration.
- Add a GitHub Action to build and push the backend Docker image to a registry.
- Finish environment variable recommendations and example `docker-compose.yml` for local testing.

What I added for you

- `docker-compose.yml` — run the backend locally with Docker for smoke tests.
- `api/.env.example` — example env vars for local dev.
- `render.yaml` — Render service definition (fill secrets in Render dashboard).
- `.github/workflows/deploy-backend.yml` — GitHub Action to build/push image to GHCR and trigger a Render deploy (set `RENDER_SERVICE_ID` and `RENDER_API_KEY` as repository secrets).

Next I can run a local Docker build and start the service, or help you configure Vercel/Render with the required secrets.

