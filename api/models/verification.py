from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from datetime import datetime
from database import Base

class VerificationRecord(Base):
    __tablename__ = "verification_records"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_path = Column(String)
    media_type = Column(String)  # 'image' or 'video'
    is_deepfake = Column(Boolean)
    confidence_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    processing_time_ms = Column(Integer)
