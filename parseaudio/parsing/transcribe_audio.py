import librosa
import soundfile as sf
from typing import List, Dict
import os
from dotenv import load_dotenv
from faster_whisper import WhisperModel

class FasterWhisperTranscriber:
    def __init__(self, model_size="large-v3", offline_mode=True):
        load_dotenv()
        self.language = "ko"
        self.sr = 16000
        
        # 모델 로드 (오프라인 모드로 설정)
        print(f"Loading Faster Whisper {model_size} model...")
        self.model = WhisperModel(
            model_size,
            device="cuda" if self._is_cuda_available() else "cpu",
            compute_type="float16" if self._is_cuda_available() else "int8",
            download_root="./models",
            local_files_only=offline_mode  # 오프라인 모드에서는 로컬 파일만 사용
        )
        print("Faster Whisper model loaded successfully")
        
    def _is_cuda_available(self):
        """CUDA 사용 가능 여부를 확인합니다."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

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
            # Faster Whisper 모델로 인식 수행 (VAD 비활성화, 빔서치 크기 증가)
            segments, _ = self.model.transcribe(
                segment_file, 
                language=self.language,
                initial_prompt=prompt,  # 이전 컨텍스트를 initial_prompt로 전달
                beam_size=5,  # 빔 서치 크기 증가
                vad_filter=False,  # VAD 비활성화
                condition_on_previous_text=True  # 이전 텍스트를 고려하여 문맥 연결성 향상
            )
            
            # 결과 텍스트 합치기
            full_text = " ".join([segment.text for segment in segments])
            
            # 결과 반환
            return full_text.strip()
        except Exception as e:
            print(f"음성 인식 오류: {str(e)}")
            return ""

    def transcribe_segments(self, audio_file: str, segments: List[Dict]) -> Dict[str, List[Dict]]:
        """세그먼트별 프롬프트 업데이트 방식으로 음성 인식 수행."""
        speaker_transcripts = {"SPK0": [], "SPK1": []}
        prompt = ""  # 프롬프트 초기화

        total_segments = len(segments)
        for idx, segment in enumerate(segments):
            print(f"처리 중: 세그먼트 {idx+1}/{total_segments}")
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
            print(f"세그먼트 {idx+1} 결과: {text[:50]}..." if len(text) > 50 else f"세그먼트 {idx+1} 결과: {text}")

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
    from diarize_speaker import SpeakerDiarizer

    # 오디오 파일 및 화자 구분 정보 불러오기
    audio_file = "audio.wav"
    diarizer = SpeakerDiarizer()
    diarization_results = diarizer.process_audio(audio_file)
    segments = diarizer.get_results_as_list(diarization_results)

    # 인식 수행 (오프라인 모드로 설정)
    transcriber = FasterWhisperTranscriber(model_size="large-v3", offline_mode=False)
    transcripts = transcriber.transcribe_segments(audio_file, segments)

    # 결과 출력
    print("\n=== 트랜스크립션 결과 ===\n")
    print(transcriber.format_conversation(transcripts))