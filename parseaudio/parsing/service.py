import os
import sys
from typing import Dict, List

from django.core.files.base import ContentFile

# voice_phishing 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parseaudio.parsing.old.transcribe_audio import WhisperTranscriber
from parseaudio.parsing.old.analyze_text import VoicePhishingDetector
from parseaudio.parsing.old.diarize_speaker import SpeakerDiarizer


diarizer = SpeakerDiarizer()
transcriber = WhisperTranscriber()
detector = VoicePhishingDetector()


def bynary_to_content_file(audio_binary, idx):
    """
    numpy.ndarray 데이터를 buffer로 변환.
    """
    if not audio_binary:
        raise ValueError("오디오 바이너리 데이터가 유효하지 않습니다.")
    # 메모리 내에서 WAV 파일 생성
    return ContentFile(audio_binary, name=f"segment_{idx}.wav")


def parse_audio(audio_file):    # 세그먼트 분할
    """Dummy audio analysis logic."""
    # SpeakerDiarizer 인스턴스 생성 및 화자 구분 수행
    diarization_results = diarizer.process_audio(audio_file)
    segments = diarizer.get_results_as_list(diarization_results)

    if not segments:
        raise ValueError("화자 구분에 실패했습니다.")

    # WhisperTranscriber 인스턴스 생성 및 세그먼트별 음성 인식 수행
    transcripts = transcriber.transcribe_segments(audio_file, segments)
    if not transcripts:
        raise ValueError("오디오 텍스트 변환에 실패했습니다.")
    return transcripts


def get_sorted_segments(speaker_transcripts: Dict[str, List[Dict]]) -> List[Dict]:
    """시간 순서대로 발화 내용을 정렬."""
    # 모든 발화를 하나의 리스트에 모은 뒤, 시작 시간을 기준으로 정렬
    all_segments = []
    for speaker, segments in speaker_transcripts.items():
        for seg in segments:
            all_segments.append({
                "speaker": speaker,
                "start": seg['start'],
                "end": seg['end'],
                "text": seg['text'],
                "audio": seg['audio'],
            })
    # 시간 순서로 정렬
    all_segments.sort(key=lambda x: x['start'])
    return all_segments


def detect_audio(transcripts):
    """음성 텍스트를 분석하여 피싱 여부 판단."""
    if not transcripts:
        raise ValueError("분석할 텍스트가 없습니다.")
    # 음성 인식 결과
    formatted_transcript = transcriber.format_conversation(transcripts)
    # VoicePhishingDetector 인스턴스 생성 및 분석 수행
    result = detector.analyze_conversation(formatted_transcript)
    return result
