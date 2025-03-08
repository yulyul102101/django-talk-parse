import os
import sys

# voice_phishing 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parseaudio.parsing.old.transcribe_audio import WhisperTranscriber
from parseaudio.parsing.old.analyze_text import VoicePhishingDetector
from parseaudio.parsing.old.diarize_speaker import SpeakerDiarizer

def test_voice_phishing_detection(audio_file_path):
    # SpeakerDiarizer 인스턴스 생성 및 화자 구분 수행
    diarizer = SpeakerDiarizer()
    diarization_results = diarizer.process_audio(audio_file_path)
    segments = diarizer.get_results_as_list(diarization_results)
    
    if not segments:
        print("화자 구분에 실패했습니다.")
        return
        
    # WhisperTranscriber 인스턴스 생성 및 세그먼트별 음성 인식 수행
    transcriber = WhisperTranscriber()
    transcripts = transcriber.transcribe_segments(audio_file_path, segments)
    
    if transcripts:
        # 음성 인식 결과 출력
        formatted_transcript = transcriber.format_conversation(transcripts)
        print("\n음성 인식 결과:")
        print(formatted_transcript)
        
        # VoicePhishingDetector 인스턴스 생성 및 분석 수행
        detector = VoicePhishingDetector()
        result = detector.analyze_conversation(formatted_transcript)
        
        print("\n보이스피싱 분석 결과:")
        print("판단:", result["judgment"])
        print("근거:", result["evidence"])
    else:
        print("음성 인식에 실패했습니다.")

if __name__ == "__main__":
    # 테스트할 오디오 파일 경로 지정
    audio_file_path = "audio.wav"
    
    # 테스트 실행
    test_voice_phishing_detection(audio_file_path)
