import os
import time
import uuid
import shutil
import cv2
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models.verification import VerificationRecord
from transformers import pipeline

router = APIRouter(
    prefix="/verify",
    tags=["verification"]
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global pipeline instance, loaded lazily
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            # Load classification pipeline using reliable deepfake detection model
            # Model: dima806/deepfake_vs_real_image_detection is fine-tuned on FF++
            _pipeline = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
        except Exception as e:
            # Log error
            print(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Could not load ML model: {str(e)}")
    return _pipeline

@router.post("/")
async def verify_media(file: UploadFile = File(...), db: Session = Depends(get_db)):
    start_time = time.time()
    
    # Secure filename
    extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

    # Determine media type
    media_type = 'video' if extension.lower() in ['.mp4', '.avi', '.mov', '.webm'] else 'image'
    
    try:
        if media_type == 'image':
            # Load model and predict
            pipe = get_pipeline()
            # Open image using PIL
            pil_img = Image.open(file_path).convert("RGB")
            results = pipe(pil_img)
            print("Model results (image):", results)  # debug log
            
            top_pred = results[0]
            label = top_pred['label'].lower()
            score = float(top_pred['score'])
            
            is_deepfake = False
            # Check labels for dima806/deepfake_vs_real_image_detection (labels are usually "real" and "fake")
            # Also handle other common label formats
            if 'fake' in label or '0' in label or label == 'deepfake':
                is_deepfake = True
            elif 'real' in label or '1' in label or label == 'authentic':
                is_deepfake = False
            else:
                is_deepfake = 'fake' in label
                
            # For models that output both classes, find the fake score
            fake_score = None
            real_score = None
            for pred in results:
                pred_label = pred['label'].lower()
                if 'fake' in pred_label or '0' in pred_label:
                    fake_score = float(pred['score'])
                if 'real' in pred_label or '1' in pred_label:
                    real_score = float(pred['score'])
            
            if fake_score is not None and real_score is not None:
                confidence_score = fake_score if is_deepfake else real_score
            else:
                # If only top result available
                confidence_score = score if is_deepfake else (1.0 - score)
            
        else: # video
            # Open video
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise ValueError("Empty video file")
                
            # We will extract 5 evenly spaced frames
            num_frames_to_check = 5
            step = max(1, frame_count // num_frames_to_check)
            
            scores = []
            fakes = 0
            
            pipe = get_pipeline()
            
            for i in range(num_frames_to_check):
                frame_idx = i * step
                if frame_idx >= frame_count:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                results = pipe(pil_img)
                print(f"Model results (video frame {i}):", results)  # debug log
                
                top_pred = results[0]
                label = top_pred['label'].lower()
                score = float(top_pred['score'])
                
                is_fake = False
                if 'fake' in label or '0' in label or label == 'deepfake':
                    is_fake = True
                elif 'real' in label or '1' in label or label == 'authentic':
                    is_fake = False
                else:
                    is_fake = 'fake' in label
                
                # For models that output both classes, find the fake score
                fake_score_frame = None
                real_score_frame = None
                for pred in results:
                    pred_label = pred['label'].lower()
                    if 'fake' in pred_label or '0' in pred_label:
                        fake_score_frame = float(pred['score'])
                    if 'real' in pred_label or '1' in pred_label:
                        real_score_frame = float(pred['score'])
                
                if fake_score_frame is not None and real_score_frame is not None:
                    frame_confidence = fake_score_frame if is_fake else real_score_frame
                else:
                    frame_confidence = score if is_fake else (1.0 - score)
                    
                scores.append(frame_confidence)
                if is_fake:
                    fakes += 1
                    
            cap.release()
            
            if not scores:
                raise ValueError("Could not extract frames from video")
                
            # Majority vote
            is_deepfake = fakes > (len(scores) // 2)
            confidence_score = sum(scores) / len(scores)
            if not is_deepfake:
                confidence_score = 1.0 - confidence_score
                
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    processing_time_ms = int((time.time() - start_time) * 1000)
    
    # Save to Database
    record = VerificationRecord(
        filename=file.filename,
        file_path=file_path,
        media_type=media_type,
        is_deepfake=is_deepfake,
        confidence_score=confidence_score,
        processing_time_ms=processing_time_ms
    )
    
    db.add(record)
    db.commit()
    db.refresh(record)
    
    return {
        "id": record.id,
        "filename": record.filename,
        "media_type": record.media_type,
        "is_deepfake": record.is_deepfake,
        "confidence_score": record.confidence_score,
        "processing_time_ms": record.processing_time_ms,
        "timestamp": record.timestamp.isoformat() if record.timestamp else None
    }

@router.get("/history")
async def get_verification_history(limit: int = 50, db: Session = Depends(get_db)):
    records = db.query(VerificationRecord).order_by(VerificationRecord.timestamp.desc()).limit(limit).all()
    return records

@router.delete("/history")
async def clear_verification_history(db: Session = Depends(get_db)):
    try:
        # Delete all records
        db.query(VerificationRecord).delete()
        db.commit()
        return {"status": "success", "message": "All verification history cleared"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")
