import os
import mimetypes
from django.core.files.storage import Storage
from django.core.files.base import ContentFile
from django.conf import settings


class VercelBlobMediaStorage(Storage):
    """Django storage class for Vercel Blob-backed media files."""

    def __init__(self):
        self.endpoint = settings.VERCEL_BLOB_ENDPOINT
        self.bucket = settings.VERCEL_BLOB_BUCKET
        self.base_url = settings.VERCEL_BLOB_BASE_URL or self.endpoint
        if not self.endpoint:
            raise ValueError('VERCEL_BLOB_ENDPOINT must be configured for Vercel blob storage')

    def _open(self, name, mode='rb'):
        raise NotImplementedError('Vercel blob storage is write-only in this implementation')

    def _save(self, name, content):
        from urllib.parse import urljoin
        import requests

        url = urljoin(self.endpoint, f'{self.bucket}/{name}')
        content.seek(0)
        headers = {
            'Content-Type': mimetypes.guess_type(name)[0] or 'application/octet-stream',
        }
        response = requests.put(url, data=content.read(), headers=headers)
        response.raise_for_status()
        return name

    def exists(self, name):
        return False

    def url(self, name):
        from urllib.parse import urljoin
        return urljoin(self.base_url, f'{self.bucket}/{name}')

    def size(self, name):
        return None

    def delete(self, name):
        pass
