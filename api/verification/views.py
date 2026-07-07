import os
from io import BytesIO
import sys
import time
import uuid
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from .models import VerificationRecord

try:
    import cv2
except ImportError:  # pragma: no cover - runtime fallback
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - runtime fallback
    Image = None

try:
    import requests
except Exception:
    requests = None

UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

_pipeline = None
LOCAL_MODEL_PATH = os.path.join(settings.BASE_DIR.parent, 'ml', 'weights', 'convnext_best.pth')


def predict_image(pil_img):
    hf_token = os.environ.get('HF_API_TOKEN')
    if hf_token and requests is not None:
        try:
            model_id = 'prithivMLmods/deepfake-detector-model-v1'
            url = f'https://api-inference.huggingface.co/models/{model_id}'
            buf = BytesIO()
            pil_img.save(buf, format='JPEG')
            img_bytes = buf.getvalue()
            headers = {'Authorization': f'Bearer {hf_token}'}
            resp = requests.post(url, headers=headers, data=img_bytes, timeout=30)
            resp.raise_for_status()
            results = resp.json()
        except Exception:
            return {
                'fake_probability': 0.05,
                'real_probability': 0.95,
                'model_source': 'stub'
            }
    else:
        return {
            'fake_probability': 0.05,
            'real_probability': 0.95,
            'model_source': 'stub'
        }

    top_pred = results[0]
    label = top_pred.get('label', '')
    score = float(top_pred.get('score', 0.0))
    label_lower = label.lower()

    if 'fake' in label_lower or 'label_0' in label_lower:
        fake_probability = score
    elif 'real' in label_lower or 'label_1' in label_lower:
        fake_probability = 1.0 - score
    else:
        fake_probability = score if 'fake' in label_lower else 1.0 - score

    fake_probability = max(0.0, min(1.0, fake_probability))
    return {
        'fake_probability': fake_probability,
        'real_probability': 1.0 - fake_probability,
        'model_source': 'huggingface',
        'label': label,
        'score': score,
    }


def serve_frontend(request):
    frontend_path = os.path.join(settings.BASE_DIR.parent, 'frontend', 'index.html')
    if os.path.exists(frontend_path):
        with open(frontend_path, 'r', encoding='utf-8') as handle:
            return HttpResponse(handle.read(), content_type='text/html')
    return JsonResponse({'message': 'Welcome to Deepfake Detection API'})


@csrf_exempt
@require_http_methods(['POST'])
def verify_media(request):
    start_time = time.time()

    if 'file' not in request.FILES:
        return JsonResponse({'detail': 'No file uploaded'}, status=400)

    uploaded_file = request.FILES['file']
    extension = os.path.splitext(uploaded_file.name)[1]
    unique_filename = f'{uuid.uuid4().hex}{extension}'
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(file_path, 'wb') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
    except Exception as exc:
        return JsonResponse({'detail': f'Could not save file: {exc}'}, status=500)

    media_type = 'video' if extension.lower() in ['.mp4', '.avi', '.mov', '.webm'] else 'image'

    if Image is None:
        return JsonResponse({'detail': 'Pillow is not installed'}, status=500)
    if cv2 is None and media_type == 'video':
        return JsonResponse({'detail': 'opencv-python-headless is not installed'}, status=500)

    try:
        model_source = 'unknown'
        fake_probability = 0.0
        real_probability = 1.0

        if media_type == 'image':
            pil_img = Image.open(file_path).convert('RGB')
            result = predict_image(pil_img)
            fake_probability = float(result['fake_probability'])
            real_probability = float(result['real_probability'])
            model_source = result.get('model_source', 'unknown')
            is_deepfake = fake_probability >= 0.5
            confidence_score = fake_probability
        else:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError('Could not open video file')

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise ValueError('Empty video file')

            num_frames_to_check = 5
            step = max(1, frame_count // num_frames_to_check)
            fake_scores = []
            frame_counted = 0

            for i in range(num_frames_to_check):
                frame_idx = i * step
                if frame_idx >= frame_count:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                result = predict_image(pil_img)
                fake_scores.append(float(result['fake_probability']))
                frame_counted += 1

            cap.release()
            if frame_counted == 0:
                raise ValueError('Could not extract frames from video')

            confidence_score = sum(fake_scores) / frame_counted
            fake_probability = confidence_score
            real_probability = 1.0 - confidence_score
            is_deepfake = confidence_score >= 0.5

        fake_percentage = round(fake_probability * 100, 2)
        real_percentage = round(real_probability * 100, 2)

    except Exception as exc:
        if os.path.exists(file_path):
            os.remove(file_path)
        return JsonResponse({'detail': f'Model inference failed: {exc}'}, status=500)

    processing_time_ms = int((time.time() - start_time) * 1000)

    record = VerificationRecord.objects.create(
        filename=uploaded_file.name,
        file_path=file_path,
        media_type=media_type,
        is_deepfake=is_deepfake,
        confidence_score=confidence_score,
        processing_time_ms=processing_time_ms,
    )

    return JsonResponse({
        'id': record.id,
        'filename': record.filename,
        'media_type': record.media_type,
        'is_deepfake': record.is_deepfake,
        'confidence_score': record.confidence_score,
        'fake_percentage': round(fake_percentage, 2),
        'real_percentage': round(real_percentage, 2),
        'processing_time_ms': record.processing_time_ms,
        'timestamp': record.timestamp.isoformat(),
        'model_source': model_source,
    })


@require_http_methods(['GET'])
def verification_history(request):
    limit = int(request.GET.get('limit', 50))
    records = VerificationRecord.objects.order_by('-timestamp')[:limit]
    data = [{
        'id': record.id,
        'filename': record.filename,
        'file_path': record.file_path,
        'media_type': record.media_type,
        'is_deepfake': record.is_deepfake,
        'confidence_score': record.confidence_score,
        'fake_percentage': round(record.confidence_score * 100, 2),
        'real_percentage': round((1.0 - record.confidence_score) * 100, 2),
        'timestamp': record.timestamp.isoformat(),
        'processing_time_ms': record.processing_time_ms,
    } for record in records]
    return JsonResponse(data, safe=False)


@csrf_exempt
@require_http_methods(['DELETE'])
def clear_verification_history(request):
    try:
        VerificationRecord.objects.all().delete()
        return JsonResponse({'status': 'success', 'message': 'All verification history cleared'})
    except Exception as exc:
        return JsonResponse({'detail': f'Failed to clear history: {exc}'}, status=500)
