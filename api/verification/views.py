import os
from io import BytesIO
import sys
import time
import uuid
import numpy as np
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import VerificationRecord


def cors_options_response():
    response = HttpResponse(status=200)
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
    response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    return response

try:
    import cv2
except ImportError:  # pragma: no cover - runtime fallback
    cv2 = None

try:
    import logging
except Exception:
    logging = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - runtime fallback
    Image = None

try:
    import torch
except Exception:  # pragma: no cover - runtime fallback
    torch = None

try:
    from torchvision import transforms
except Exception:  # pragma: no cover - runtime fallback
    transforms = None

if logging is not None:
    logger = logging.getLogger(__name__)
else:
    class _DummyLogger:
        def info(self, *a, **k):
            pass
        def warning(self, *a, **k):
            pass
        def exception(self, *a, **k):
            pass
    logger = _DummyLogger()
try:
    import requests
except Exception:
    requests = None

UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

_pipeline = None
LOCAL_MODEL_PREFERRED_PATH = os.path.join(settings.BASE_DIR.parent, 'ml', 'weights', 'convnext_large_best.pth')
LOCAL_MODEL_FALLBACK_PATH = os.path.join(settings.BASE_DIR.parent, 'ml', 'weights', 'convnext_best.pth')
_LOCAL_MODEL = None


def run_local_inference(pil_img):
    if torch is None or transforms is None:
        raise RuntimeError('PyTorch and torchvision are required for local inference')

    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        model_path = LOCAL_MODEL_PREFERRED_PATH if os.path.exists(LOCAL_MODEL_PREFERRED_PATH) else LOCAL_MODEL_FALLBACK_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'No local checkpoint found at {LOCAL_MODEL_PREFERRED_PATH} or {LOCAL_MODEL_FALLBACK_PATH}'
            )

        # Prefer loading the model class directly from the ml/models file to avoid
        # import shadowing with other `models` packages in the project.
        convnext_file = os.path.join(settings.BASE_DIR.parent, 'ml', 'models', 'convnext_detector.py')
        if os.path.exists(convnext_file):
            import importlib.util
            spec = importlib.util.spec_from_file_location('ml_convnext_detector', convnext_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ConvNeXtDetector = getattr(module, 'ConvNeXtDetector')
            backbone_name = 'convnext_large' if 'convnext_large' in os.path.basename(model_path) else 'convnext_base'
            model = ConvNeXtDetector(pretrained=False, backbone_name=backbone_name)
        else:
            # Fallback to regular import (may fail if package shadowing occurs)
            sys.path.append(os.path.join(settings.BASE_DIR.parent, 'ml'))
            from models.convnext_detector import ConvNeXtDetector
            backbone_name = 'convnext_large' if 'convnext_large' in os.path.basename(model_path) else 'convnext_base'
            model = ConvNeXtDetector(pretrained=False, backbone_name=backbone_name)
        state = torch.load(model_path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state_dict = state['state_dict']
        else:
            state_dict = state
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        _LOCAL_MODEL = model

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = preprocess(pil_img.convert('RGB')).unsqueeze(0)
    with torch.inference_mode():
        output = _LOCAL_MODEL(image_tensor)

    if output.ndim > 0:
        score = float(torch.sigmoid(output).detach().cpu().squeeze().item())
    else:
        score = float(torch.sigmoid(output).detach().cpu().item())

    fake_probability = max(0.0, min(1.0, float(score)))
    return {
        'fake_probability': fake_probability,
        'real_probability': 1.0 - fake_probability,
        'model_source': 'local',
    }


def _normalize_score(score):
    return max(0.0, min(1.0, float(score)))


def _compute_frequency_artifact_score(pil_img):
    img_gray = np.asarray(pil_img.convert('L'), dtype=np.float32)
    if img_gray.size == 0:
        return 0.0
    freq = np.fft.fftshift(np.fft.fft2(img_gray))
    magnitude = np.abs(freq)
    h, w = img_gray.shape
    cy, cx = h // 2, w // 2
    radius = max(8, min(h, w) // 8)
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    low_energy = magnitude[mask].sum()
    total_energy = magnitude.sum() + 1e-9
    high_ratio = 1.0 - (low_energy / total_energy)
    return _normalize_score((high_ratio - 0.30) * 2.0)


def _compute_noise_consistency_score(pil_img):
    img_gray = np.asarray(pil_img.convert('L'), dtype=np.float32)
    if img_gray.size == 0:
        return 0.0
    if cv2 is not None:
        blurred = cv2.GaussianBlur(img_gray, (7, 7), 0)
    else:
        blurred = img_gray
    noise = img_gray - blurred
    h, w = img_gray.shape
    tile = 32
    stds = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            patch = noise[y:y+tile, x:x+tile]
            if patch.size:
                stds.append(np.std(patch))
    if not stds:
        return 0.0
    mean_std = float(np.mean(stds)) + 1e-9
    variation = float(np.std(stds)) / mean_std
    score = (variation - 0.25) * 1.5
    return _normalize_score(score)


def _compute_compression_anomaly_score(pil_img):
    try:
        buf = BytesIO()
        pil_img.save(buf, format='JPEG', quality=85)
        buf.seek(0)
        recompressed = Image.open(buf).convert('RGB')
        original = np.asarray(pil_img.convert('RGB'), dtype=np.float32)
        recreated = np.asarray(recompressed, dtype=np.float32)
        diff = np.abs(original - recreated)
        err = float(np.mean(diff)) / 255.0
        return _normalize_score((err - 0.015) * 3.0)
    except Exception:
        return 0.0


def _compute_face_integrity_score(pil_img):
    if cv2 is None:
        return 0.0
    try:
        rgb = np.asarray(pil_img.convert('RGB'), dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if face_cascade.empty():
            return 0.0
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return 0.2

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = gray[y:y+h, x:x+w]
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        eyes = []
        if not eye_cascade.empty():
            eyes = eye_cascade.detectMultiScale(face, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))

        eye_score = min(1.0, len(eyes) / 2.0) if len(eyes) > 0 else 0.0
        aspect_ratio = float(w) / float(h + 1e-9)
        aspect_score = _normalize_score(1.0 - abs(aspect_ratio - 0.85) * 2.5)
        edge = cv2.Canny(face, 100, 200)
        border_strength = (float(np.sum(edge[:4, :])) + np.sum(edge[-4:, :]) + np.sum(edge[:, :4]) + np.sum(edge[:, -4:])) / (face.size + 1)
        boundary_score = _normalize_score(1.0 - border_strength * 4.0)

        return _normalize_score(0.35 + 0.35 * eye_score + 0.15 * aspect_score + 0.15 * boundary_score)
    except Exception:
        return 0.0


def _compute_texture_coherence_score(pil_img):
    img_gray = np.asarray(pil_img.convert('L'), dtype=np.uint8)
    if img_gray.size == 0 or cv2 is None:
        return 0.0
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    lap_var = float(np.var(lap))
    if lap_var < 120.0:
        return _normalize_score((120.0 - lap_var) / 120.0)
    return 0.0


def _compute_metadata_score(pil_img):
    try:
        exif = pil_img._getexif() if hasattr(pil_img, '_getexif') else None
        if not exif or len(exif) == 0:
            return 0.25
        suspicious_tags = ['Software', 'Make', 'Model']
        if any(str(exif.get(tag, '')).lower().find('adobe') >= 0 or str(exif.get(tag, '')).lower().find('iphone') >= 0 for tag in suspicious_tags):
            return 0.0
        return 0.0
    except Exception:
        return 0.25


def _compute_gan_artifact_score(pil_img):
    frequency = _compute_frequency_artifact_score(pil_img)
    noise = _compute_noise_consistency_score(pil_img)
    compression = _compute_compression_anomaly_score(pil_img)
    return _normalize_score((frequency + noise + compression) / 3.0)


def _compute_artifact_scores(pil_img):
    face_integrity = _compute_face_integrity_score(pil_img)
    texture_coh = _compute_texture_coherence_score(pil_img)
    gan_artifact = _compute_gan_artifact_score(pil_img)
    frequency_artifact = _compute_frequency_artifact_score(pil_img)
    noise_consistency = _compute_noise_consistency_score(pil_img)
    compression_anomaly = _compute_compression_anomaly_score(pil_img)
    metadata = _compute_metadata_score(pil_img)
    return {
        'face_integrity': face_integrity,
        'gan_artifact': gan_artifact,
        'texture_coh': texture_coh,
        'frequency_artifact': frequency_artifact,
        'noise_consistency': noise_consistency,
        'compression_anomaly': compression_anomaly,
        'metadata': metadata,
    }


def _combine_model_and_artifacts(model_score, component_scores):
    artifact_evidence = float(np.mean([
        component_scores.get('face_integrity', 0.0),
        component_scores.get('gan_artifact', 0.0),
        component_scores.get('texture_coh', 0.0),
    ]))
    artifact_score = float(np.mean(list(component_scores.values())))
    combined = _normalize_score(model_score * 0.55 + artifact_evidence * 0.45)
    return combined, artifact_score, artifact_evidence


def predict_image(pil_img):
    artifact_scores = _compute_artifact_scores(pil_img)
    if callable(run_local_inference):
        try:
            local_result = run_local_inference(pil_img)
            model_score = float(local_result.get('fake_probability', 0.0))
            combined_score, artifact_score, artifact_evidence = _combine_model_and_artifacts(model_score, artifact_scores)
            return {
                'fake_probability': combined_score,
                'real_probability': 1.0 - combined_score,
                'model_source': 'local',
                'artifact_score': artifact_score,
                'artifact_evidence': artifact_evidence,
                'artifact_components': artifact_scores,
                **local_result,
            }
        except Exception as exc:
            logger.warning('Local inference failed, falling back to stub: %s', exc)

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

    fake_probability = _normalize_score(fake_probability)
    combined_score, artifact_score, artifact_evidence = _combine_model_and_artifacts(fake_probability, artifact_scores)
    return {
        'fake_probability': combined_score,
        'real_probability': 1.0 - combined_score,
        'model_source': 'huggingface',
        'artifact_score': artifact_score,
        'artifact_evidence': artifact_evidence,
        'artifact_components': artifact_scores,
        'label': label,
        'score': score,
    }


def _map_score_to_verdict(fake_percentage: float, artifact_components=None):
    """Map a score (0-100) to a verdict, adapting thresholds based on artifacts.

    Higher artifact evidence lowers the boundary for PARTIALLY_REAL/FAKE.
    Lower artifact evidence raises the boundary for REAL.
    """
    try:
        p = float(fake_percentage)
    except Exception:
        return 'UNKNOWN'

    artifact_components = artifact_components or {}
    if artifact_components:
        artifact_score = float(np.mean(list(artifact_components.values())))
    else:
        artifact_score = 0.0

    if artifact_score >= 0.6:
        fake_threshold = 45.0
        partial_threshold = 68.0
    elif artifact_score <= 0.25:
        fake_threshold = 55.0
        partial_threshold = 87.0
    else:
        fake_threshold = 50.0
        partial_threshold = 80.0

    if p < fake_threshold:
        return 'FAKE'
    if p < partial_threshold:
        return 'PARTIALLY_REAL'
    return 'REAL'


def serve_frontend(request):
    frontend_path = os.path.join(settings.BASE_DIR.parent, 'frontend', 'index.html')
    if os.path.exists(frontend_path):
        with open(frontend_path, 'r', encoding='utf-8') as handle:
            return HttpResponse(handle.read(), content_type='text/html')
    return JsonResponse({'message': 'Welcome to Deepfake Detection API'})


@csrf_exempt
def verify_media(request):
    if request.method == 'OPTIONS':
        return cors_options_response()
    if request.method != 'POST':
        return JsonResponse({'detail': 'Method not allowed'}, status=405)
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

    # If Pillow isn't installed, bail early for image uploads to avoid calling Image.open
    if media_type == 'image' and Image is None:
        if os.path.exists(file_path):
            os.remove(file_path)
        return JsonResponse({'detail': 'Pillow is not installed'}, status=500)

    # Basic validation: ensure uploaded file is a valid image or video so we
    # return 400 (client error) instead of letting inference raise a 500.
    try:
        logger.info("Received uploaded file %s, detected media_type=%s", uploaded_file.name, media_type)
        if media_type == 'image':
            try:
                pil_test = Image.open(file_path)
                pil_test.verify()
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                logger.warning("Invalid image upload saved at %s: %s", file_path, e)
                return JsonResponse({'detail': 'Uploaded file is not a valid image'}, status=400)

        else:  # video
            if cv2 is None:
                if os.path.exists(file_path):
                    os.remove(file_path)
                logger.warning("Received video upload but OpenCV not installed: %s", file_path)
                return JsonResponse({'detail': 'Video processing not available on server'}, status=400)
            cap_test = cv2.VideoCapture(file_path)
            if not cap_test.isOpened():
                cap_test.release()
                if os.path.exists(file_path):
                    os.remove(file_path)
                logger.warning("Uploaded video could not be opened: %s", file_path)
                return JsonResponse({'detail': 'Uploaded file is not a valid video'}, status=400)
            ret, frame = cap_test.read()
            cap_test.release()
            if not ret:
                if os.path.exists(file_path):
                    os.remove(file_path)
                logger.warning("Uploaded video has no readable frames: %s", file_path)
                return JsonResponse({'detail': 'Uploaded video contains no readable frames'}, status=400)
    except Exception:
        # Any unexpected validation error should not leak internals.
        logger.exception("Unexpected error during upload validation for %s", file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        return JsonResponse({'detail': 'Invalid upload'}, status=400)

    if Image is None:
        return JsonResponse({'detail': 'Pillow is not installed'}, status=500)
    if cv2 is None and media_type == 'video':
        return JsonResponse({'detail': 'opencv-python-headless is not installed'}, status=500)

    try:
        model_source = 'unknown'
        logger.info("Starting model inference for %s (media_type=%s)", file_path, media_type)
        fake_probability = 0.0
        real_probability = 1.0

        if media_type == 'image':
            pil_img = Image.open(file_path).convert('RGB')
            result = predict_image(pil_img)
            fake_probability = float(result['fake_probability'])
            real_probability = float(result['real_probability'])
            model_source = result.get('model_source', 'unknown')
            artifact_score = result.get('artifact_score', 0.0)
            artifact_evidence = result.get('artifact_evidence', 0.0)
            artifact_components = result.get('artifact_components', {})
            is_deepfake = fake_probability >= 0.5
            confidence_score = fake_probability
        else:
            if cv2 is None:
                raise ValueError('opencv-python-headless is not installed')

            cv2_module = cv2
            cap = cv2_module.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError('Could not open video file')

            frame_count = int(cap.get(cv2_module.CAP_PROP_FRAME_COUNT))
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
                cap.set(cv2_module.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
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

        # Map score to qualitative verdicts per user thresholds
        verdict = _map_score_to_verdict(fake_percentage, artifact_components if media_type == 'image' else None)

    except Exception as exc:
        logger.exception("Model inference failed for %s", file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        return JsonResponse({'detail': f'Model inference failed: {str(exc)}'}, status=500)

    processing_time_ms = int((time.time() - start_time) * 1000)

    record = VerificationRecord.objects.create(
        filename=uploaded_file.name,
        file_path=file_path,
        media_type=media_type,
        is_deepfake=is_deepfake,
        confidence_score=confidence_score,
        processing_time_ms=processing_time_ms,
    )

    response_data = {
        'id': record.pk,
        'filename': record.filename,
        'media_type': record.media_type,
        'is_deepfake': record.is_deepfake,
        'confidence_score': record.confidence_score,
        'fake_percentage': round(fake_percentage, 2),
        'real_percentage': round(real_percentage, 2),
        'processing_time_ms': record.processing_time_ms,
        'timestamp': record.timestamp.isoformat(),
        'model_source': model_source,
        'verdict': verdict,
    }
    if media_type == 'image':
        response_data['artifact_score'] = artifact_score
        response_data['artifact_evidence'] = artifact_evidence
        response_data['artifact_components'] = artifact_components
        response_data['face_integrity'] = artifact_components.get('face_integrity')
        response_data['gan_artifact'] = artifact_components.get('gan_artifact')
        response_data['texture_coh'] = artifact_components.get('texture_coh')

    return JsonResponse(response_data)


def verification_history(request):
    if request.method == 'OPTIONS':
        return cors_options_response()
    if request.method != 'GET':
        return JsonResponse({'detail': 'Method not allowed'}, status=405)
    limit = int(request.GET.get('limit', 50))
    records = VerificationRecord.objects.order_by('-timestamp')[:limit]
    data = [{
        'id': record.pk,
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
def verify_compare(request):
    if request.method == 'OPTIONS':
        return cors_options_response()
    if request.method != 'POST':
        return JsonResponse({'detail': 'Method not allowed'}, status=405)

    if 'file1' not in request.FILES or 'file2' not in request.FILES:
        return JsonResponse({'detail': 'Both file1 and file2 must be uploaded'}, status=400)

    # Require Pillow for image operations
    if Image is None:
        return JsonResponse({'detail': 'Pillow is not installed'}, status=500)

    file1 = request.FILES['file1']
    file2 = request.FILES['file2']

    def _save_uploaded(f):
        ext = os.path.splitext(f.name)[1]
        name = f'{uuid.uuid4().hex}{ext}'
        path = os.path.join(UPLOAD_DIR, name)
        with open(path, 'wb') as dest:
            for chunk in f.chunks():
                dest.write(chunk)
        return path

    path1 = _save_uploaded(file1)
    path2 = _save_uploaded(file2)

    try:
        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')

        r1 = predict_image(img1)
        r2 = predict_image(img2)

        p1 = round(float(r1.get('fake_probability', 0.0)) * 100, 2)
        p2 = round(float(r2.get('fake_probability', 0.0)) * 100, 2)

        verdict1 = _map_score_to_verdict(p1)
        verdict2 = _map_score_to_verdict(p2)

        # Similarity via mean absolute difference on grayscale resized images
        try:
            a1 = np.array(img1.resize((256, 256)).convert('L'), dtype=np.float32)
            a2 = np.array(img2.resize((256, 256)).convert('L'), dtype=np.float32)
            mad = float(np.mean(np.abs(a1 - a2)))
            sim_mad = max(0.0, 1.0 - (mad / 255.0))
            similarity_mad_percent = round(sim_mad * 100, 2)
        except Exception:
            similarity_mad_percent = None

        # Histogram correlation via OpenCV if available
        hist_corr = None
        if cv2 is not None:
            try:
                cv_img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
                cv_img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
                corrs = []
                for ch in range(3):
                    h1 = cv2.calcHist([cv_img1], [ch], None, [256], [0, 256])
                    h2 = cv2.calcHist([cv_img2], [ch], None, [256], [0, 256])
                    cv2.normalize(h1, h1)
                    cv2.normalize(h2, h2)
                    corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
                    corrs.append(corr)
                # Average correlation (typically in range [-1,1] but usually [0,1] for images)
                hist_corr = round((sum(corrs) / len(corrs)) * 100, 2)
            except Exception:
                hist_corr = None

        response = {
            'image1': {
                'filename': file1.name,
                'fake_percentage': p1,
                'verdict': verdict1,
                'model_source': r1.get('model_source'),
            },
            'image2': {
                'filename': file2.name,
                'fake_percentage': p2,
                'verdict': verdict2,
                'model_source': r2.get('model_source'),
            },
            'similarity_mad_percent': similarity_mad_percent,
            'histogram_correlation_percent': hist_corr,
        }

    except Exception as exc:
        logger.exception('Comparison failed')
        response = {'detail': f'Comparison failed: {exc}'}
    finally:
        try:
            if os.path.exists(path1):
                os.remove(path1)
        except Exception:
            pass
        try:
            if os.path.exists(path2):
                os.remove(path2)
        except Exception:
            pass

    return JsonResponse(response)


@csrf_exempt
def clear_verification_history(request):
    if request.method == 'OPTIONS':
        return cors_options_response()
    if request.method != 'DELETE':
        return JsonResponse({'detail': 'Method not allowed'}, status=405)
    try:
        VerificationRecord.objects.all().delete()
        return JsonResponse({'status': 'success', 'message': 'All verification history cleared'})
    except Exception as exc:
        return JsonResponse({'detail': f'Failed to clear history: {exc}'}, status=500)
