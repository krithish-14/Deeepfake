
import os
import time
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models.verification import VerificationRecord

router = APIRouter(
    prefix="/verify",
    tags=["verification"]
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    # Mock analysis result for testing
    import random
    is_deepfake = random.choice([True, False])
    confidence_score = random.uniform(0.7, 0.99)

    processing_time_ms = int((time.time() - start_time) * 1000)

    # Save to Database
    record = VerificationRecord(
        filename=file.filename,
        file_path=file_path,
        media_type="video" if extension.lower() in ['.mp4', '.avi', '.mov', '.webm'] else "image",
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
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {e}")
