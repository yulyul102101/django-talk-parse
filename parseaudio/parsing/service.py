import os
import sys
from typing import Dict, List

from django.core.files.base import ContentFile
import uuid

# voice_phishing 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parseaudio.parsing.transcribe_audio import LocalFasterWhisperTranscriber
from parseaudio.parsing.analyze_text import VoicePhishingDetector
from parseaudio.parsing.diarize_speaker import SpeakerDiarizer


transcriber = LocalFasterWhisperTranscriber()


def bynary_to_content_file(audio_binary, idx):
    """
    numpy.ndarray 데이터를 buffer로 변환.
    """
    if not audio_binary:
        raise ValueError("오디오 바이너리 데이터가 유효하지 않습니다.")
    # 메모리 내에서 WAV 파일 생성
    return ContentFile(audio_binary, name=f"segment_{idx}.wav")


def parse_audio(audio_file, num_speakers):    # 세그먼트 분할
    """Dummy audio analysis logic."""
    # SpeakerDiarizer 인스턴스 생성 및 화자 구분 수행
    diarizer = SpeakerDiarizer(num_speakers=num_speakers)
    diarization_results = diarizer.process_audio(audio_file)
    segments = diarizer.get_results_as_list(diarization_results)

    if not segments:
        raise ValueError("화자 구분에 실패했습니다.")

    request_id = str(uuid.uuid4())
    # transcribe_segments 호출 시 request_id 전달
    transcripts = transcriber.transcribe_segments(audio_file, segments, request_id)
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


def get_text_only_transcripts(transcripts):
    text_only_transcripts = []
    for speaker, segments in transcripts.items():
        for seg in segments:
            text_only_transcripts.append({
                "speaker": speaker,
                "text": seg["text"],
                "start": seg["start"]  # 정렬용으로만 사용
            })

    # 시간 순 정렬
    text_only_transcripts.sort(key=lambda x: x["start"])

    # 정렬된 후 start 제거
    for seg in text_only_transcripts:
        del seg["start"]

    return text_only_transcripts


def detect_audio(transcripts, model_name):
    """음성 텍스트를 분석하여 피싱 여부 판단."""
    if not transcripts:
        raise ValueError("분석할 텍스트가 없습니다.")
    detector = VoicePhishingDetector(model_name=model_name)
    # 음성 인식 결과
    text_only_transcripts = get_text_only_transcripts(transcripts)
    result = detector.analyze_conversation(text_only_transcripts)
    # print("transcripts: ", text_only_transcripts)
    return result
