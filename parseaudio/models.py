import os
from datetime import datetime
from io import BytesIO
import soundfile as sf

from django.core.files.base import ContentFile
from django.db import models

class OriginalAudio(models.Model):
    PHISHING_CHOICES = [
        (1, '보이스피싱'),
        (0, '정상 통화'),
        (-1, '판단 어려움'),
    ]

    # 업로드 날짜를 기준으로 경로 생성
    def original_audio_path(instance, filename):
        today = datetime.now()
        return os.path.join("original", today.strftime("%Y/%m/%d"), filename)

    audio_file = models.FileField(upload_to=original_audio_path)  # 받은 full wav
    create_date = models.DateTimeField(auto_now_add=True)  # 업로드 날짜 자동 설정
    is_phishing = models.IntegerField(choices=PHISHING_CHOICES, null=True)  # 보이스피싱 여부
    phishing_reason = models.TextField(null=True, blank=True)  # 판단 근거

    # save 오버라이드
    def save(self, *args, **kwargs):
        if isinstance(self.is_phishing, str):
            if self.is_phishing == "보이스피싱":
                self.is_phishing = 1
            elif self.is_phishing == "정상 통화":
                self.is_phishing = 0
            else:
                self.is_phishing = -1  # 판단 어려움

        super().save(*args, **kwargs)  # 부모 클래스의 save 호출


class SegmentAudio(models.Model):
    # 원본 파일 업로드 날짜 + 원본 오디오 ID를 포함한 경로 생성
    def segment_audio_path(instance, filename):
        original_audio = instance.original_id  # 연결된 OriginalAudio 객체
        return os.path.join(
            "segments",
            original_audio.create_date.strftime("%Y/%m/%d"),
            f"{original_audio.id}",
            f"segment_{instance.segment_index}.wav"
        )

    original_id = models.ForeignKey(OriginalAudio, on_delete=models.CASCADE, related_name="segments")
    audio_file = models.FileField(upload_to=segment_audio_path)  # 쪼개진 segment 저장 경로
    segment_index = models.IntegerField()  # 세그먼트 번호
    speaker = models.CharField(max_length=10)  # 화자 정보
    transcription = models.TextField()  # 전사 텍스트
    start_time = models.FloatField()  # 세그먼트 시작 시간
    end_time = models.FloatField()  # 세그먼트 종료 시간