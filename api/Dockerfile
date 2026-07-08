FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc libsndfile1 libgl1 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV DJANGO_SETTINGS_MODULE=api.settings

RUN python manage.py collectstatic --noinput || true

CMD ["sh", "-c", "python manage.py migrate --noinput && python manage.py collectstatic --noinput || true && gunicorn api.wsgi:application --bind 0.0.0.0:${PORT:-8000} --workers 2"]
