import librosa
import soundfile as sf
from typing import List, Dict
import os
import glob
import subprocess
from dotenv import load_dotenv
import torch
from faster_whisper import WhisperModel
from huggingface_hub import login
import sys
import uuid
import shutil

def find_cudnn_version():
    """시스템에 설치된 CUDNN 버전을 찾아 반환합니다."""
    try:
        # NVIDIA driver 정보 출력
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
        print(f"NVIDIA Driver 정보:\n{result.stdout}")

        # CUDNN 라이브러리 검색
        cudnn_paths = []
        for path in ['/usr/lib/x86_64-linux-gnu/', '/usr/local/cuda/lib64/',
                     '/usr/local/cuda/targets/x86_64-linux/lib/']:
            if os.path.exists(path):
                cudnn_files = glob.glob(f"{path}libcudnn*.so*")
                if cudnn_files:
                    cudnn_paths.extend(cudnn_files)

        if cudnn_paths:
            print(f"발견된 CUDNN 라이브러리:\n{cudnn_paths}")
            return "found"
        else:
            print("CUDNN 라이브러리를 찾을 수 없습니다.")
            return "not_found"
    except Exception as e:
        print(f"CUDNN 버전 확인 오류: {str(e)}")
        return "error"

class LocalFasterWhisperTranscriber:
    def __init__(self, model_size="large-v2", device="auto", compute_type=None):
        try:
            import torch
            from faster_whisper import WhisperModel
            
            # compute_type 자동 설정
            if compute_type is None:
                # CPU인 경우 또는 GPU가 없는 경우 int8 사용
                if device == "cpu" or (device == "auto" and not torch.cuda.is_available()):
                    compute_type = "int8"  # CPU에서는 int8 사용
                else:
                    compute_type = "float16"  # GPU에서는 float16 사용
            
            print(f"사용 중인 장치: {device}, 연산 타입: {compute_type}")
            
            # 패키징된 앱인 경우 경로 설정
            if getattr(sys, 'frozen', False):
                # 실행 파일 기준 경로
                base_dir = os.path.dirname(sys.executable)
                model_path = os.path.join(base_dir, "F_Model", "whisper", model_size)
                
                # 모델 경로가 존재하는지 확인
                if os.path.exists(model_path):
                    print(f"로컬 모델 사용: {model_path}")
                    self.model = WhisperModel(
                        model_path, 
                        device=device,
                        compute_type=compute_type,
                        local_files_only=True
                    )
                else:
                    # 모델이 없으면 기본 방식으로 로드 시도
                    print(f"로컬 모델을 찾을 수 없어 기본 방식으로 로드합니다.")
                    self.model = WhisperModel(
                        model_size, 
                        device=device,
                        compute_type=compute_type
                    )
            else:
                # 일반 실행 시 기본 방식으로 로드
                self.model = WhisperModel(
                    model_size, 
                    device=device,
                    compute_type=compute_type
                )
                
            print(f"Faster Whisper 모델 '{model_size}' 로드됨")
        except ImportError:
            raise ImportError("'pip install faster-whisper'를 실행하세요.")
            
        self.language = "ko"
        self.sr = 16000

    def segment_audio(self, audio_file: str, start: float, end: float, output_file: str) -> tuple:
        """주어진 구간의 오디오를 잘라서 저장합니다."""
        y, sr = librosa.load(audio_file, sr=self.sr)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        sf.write(output_file, y[start_sample:end_sample], sr)
        return output_file, y[start_sample:end_sample], start_sample, end_sample

    def transcribe_segment(self, segment_file: str, prompt: str = "") -> str:
        """세그먼트별 음성 인식 수행 (지원되는 파라미터만 사용)."""
        try:
            segments, _ = self.model.transcribe(
                segment_file,
                language=self.language,
                initial_prompt=prompt,
                beam_size=5,             # 정확도와 속도의 균형을 위한 빔 크기[4]
                temperature=0.0,         # 확정적인 결과를 위한 온도 설정[3]
                best_of=5,               # 비-0 온도로 샘플링할 때 후보 수[3]
                no_speech_threshold=0.6  # 묵음 감지 임계값[3]
            )
            
            # 세그먼트에서 텍스트 추출
            text = " ".join([seg.text for seg in segments])
            return text
                
        except Exception as e:
            print(f"음성 인식 오류: {str(e)}")
            return ""

    def transcribe_segments(self, audio_file: str, segments: List[Dict], request_id: str) -> Dict[str, List[Dict]]:
        """세그먼트별 프롬프트 업데이트 방식으로 음성 인식 수행."""
        speaker_transcripts = {"SPK0": [], "SPK1": []}
        prompt = ""  # 프롬프트 초기화
        
        # 요청별 고유 디렉토리 생성
        output_dir = f"segments_{request_id}"
        os.makedirs(output_dir, exist_ok=True)

        for idx, segment in enumerate(segments):
            # 고유한 세그먼트 파일 이름 생성
            segment_file = os.path.join(output_dir, f"{request_id}_segment_{idx}_{segment['speaker']}.wav")
            
            # 세그먼트 파일 생성
            _, y, start_sample, end_sample = self.segment_audio(audio_file, segment['start'], segment['end'], segment_file)
            
            # 이전 프롬프트 반영하여 세그먼트 인식
            text = self.transcribe_segment(segment_file, prompt=prompt)

            # 프롬프트 누적
            prompt += " " + text  # 이전 결과 추가
            if len(prompt) > 200:  # 200자를 초과하면
                prompt = prompt[-200:]  # 뒤에서 200자만 유지

            # 결과 저장
            with open(segment_file, "rb") as file:
                audio_binary = file.read()

            speaker_transcripts[segment['speaker']].append({
                "start": segment['start'],
                "end": segment['end'],
                "text": text,
                "audio": audio_binary  # 해당 구간의 오디오 데이터 저장
            })

            # 세그먼트 파일 즉시 삭제 (성능 향상을 위해)
            os.remove(segment_file)

        # 처리 완료 후 디렉토리 삭제
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
        except:
            pass  # 디렉토리 삭제 실패해도 계속 진행

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
    audio_file ="./audio.wav"
    diarizer = SpeakerDiarizer(num_speakers=2)  # 화자 수 지정
    diarization_results = diarizer.process_audio(audio_file)
    segments = diarizer.get_results_as_list(diarization_results)

    # 인식 수행 - 모델 크기와 장치 타입 선택 가능
    transcriber = LocalFasterWhisperTranscriber(
        model_size="large-v2",  # 속도와 정확도 균형을 위해 medium 권장
        device="auto",        # GPU 있으면 자동 감지
        compute_type="float16"  # 속도 향상을 위한 양자화
    )
    request_id = str(uuid.uuid4())  # 32자리 16진수 문자열 생성

    # 성능 개선된 메서드 호출
    transcripts = transcriber.transcribe_segments(audio_file, segments, request_id)

    # 결과 출력
    print(transcriber.format_conversation(transcripts))