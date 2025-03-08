import librosa
import soundfile as sf
from typing import List, Dict
import os
from dotenv import load_dotenv
import openai

# 2024.12.22 수정됨: transcribe_segments의

class WhisperTranscriber:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "whisper-1"
        self.language = "ko"
        self.sr = 16000

    def segment_audio(self, audio_file: str, start: float, end: float, output_file: str) -> str:
        """주어진 구간의 오디오를 잘라서 저장합니다."""
        y, sr = librosa.load(audio_file, sr=self.sr)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        sf.write(output_file, y[start_sample:end_sample], sr)
        return output_file, y[start_sample:end_sample], start_sample, end_sample

    def transcribe_segment(self, segment_file: str, prompt: str = "") -> str:
        """세그먼트별 음성 인식 수행 (이전 프롬프트 활용)."""
        try:
            with open(segment_file, "rb") as audio:
                transcript = openai.Audio.transcribe(
                    model=self.model,
                    file=audio,
                    language=self.language,
                    prompt=prompt  # 이전 구간 결과를 프롬프트로 전달
                )
            return transcript.text
        except Exception as e:
            print(f"음성 인식 오류: {str(e)}")
            return ""

    def transcribe_segments(self, audio_file: str, segments: List[Dict]) -> Dict[str, List[Dict]]:
        """세그먼트별 프롬프트 업데이트 방식으로 음성 인식 수행."""
        #speaker_transcripts = {"발신자": [], "수신자": []}
        speaker_transcripts = {"SPK0": [], "SPK1": []}
        prompt = ""  # 프롬프트 초기화

        for idx, segment in enumerate(segments):
            # 세그먼트 자르기
            segment_file = f"segment_{idx}.wav"
            _, y, start_sample, end_sample = self.segment_audio(audio_file, segment['start'], segment['end'], segment_file)

            # 이전 프롬프트 반영하여 세그먼트 인식
            text = self.transcribe_segment(segment_file, prompt=prompt)

            # segment_file을 읽어 바이너리 데이터로 변환
            with open(segment_file, "rb") as file:
                audio_binary = file.read()

            os.remove(segment_file)

            # 프롬프트 누적
            prompt += " " + text  # 이전 결과 추가
            if len(prompt) > 200:  # 200자를 초과하면
                prompt = prompt[-200:]  # 뒤에서 200자만 유지

            # 결과 저장
            speaker_transcripts[segment['speaker']].append({
                "start": segment['start'],
                "end": segment['end'],
                "text": text,
                "audio": audio_binary  # 해당 구간의 오디오 데이터 저장
            })

        return speaker_transcripts

    def format_conversation(self, speaker_transcripts: Dict[str, List[Dict]]) -> str:
        """시간 순서대로 발화 내용을 포맷팅."""
        # 모든 발화를 하나의 리스트에 모은 뒤, 시작 시간을 기준으로 정렬
        all_segments = []
        for speaker, segments in speaker_transcripts.items():
            for seg in segments:
                all_segments.append({
                    "speaker": speaker,
                    "start": seg['start'],
                    "end": seg['end'],
                    "text": seg['text']
                })

        # 시간 순서로 정렬
        all_segments.sort(key=lambda x: x['start'])

        # 포맷팅
        formatted_text = ""
        for seg in all_segments:
            formatted_text += f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}: {seg['text']}\n"
        return formatted_text

if __name__ == "__main__":
    from parseaudio.parsing.old.diarize_speaker import SpeakerDiarizer

    # 오디오 파일 및 화자 구분 정보 불러오기
    audio_file = "../test/audio.wav"
    diarizer = SpeakerDiarizer()
    diarization_results = diarizer.process_audio(audio_file)
    segments = diarizer.get_results_as_list(diarization_results)

    # 인식 수행
    transcriber = WhisperTranscriber()
    transcripts = transcriber.transcribe_segments(audio_file, segments)

    # 결과 출력
    print(transcriber.format_conversation(transcripts))