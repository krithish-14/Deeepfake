from django.db import models


class VerificationRecord(models.Model):
    filename = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    media_type = models.CharField(max_length=20)
    is_deepfake = models.BooleanField()
    confidence_score = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    processing_time_ms = models.IntegerField()

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return self.filename
