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
    def __init__(self, model_size="large-v3", hf_token=None, force_cpu=False):
        """
        로컬 Faster Whisper 모델을 사용하는 음성 인식기 초기화

        Args:
            model_size: 사용할 Whisper 모델 크기 ("tiny", "base", "small", "medium", "large-v2", "large-v3")
            hf_token: Hugging Face 토큰 (비공개 모델 사용 시 필요)
            force_cpu: CPU 모드 강제 사용 여부
        """
        load_dotenv()
        self.sr = 16000
        self.language = "ko"

        # HuggingFace 토큰이 제공되면 로그인
        if hf_token:
            login(token=hf_token)
            print("Logged in to Hugging Face")

        # GPU 정보 출력 및 CUDNN 버전 확인
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            print(f"CUDA 사용 가능. 디바이스 수: {torch.cuda.device_count()}")
            print(f"CUDA 버전: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

            # CUDNN 버전 확인
            cudnn_status = find_cudnn_version()
        else:
            cudnn_status = "no_cuda"

        # 디바이스 선택
        if force_cpu or cudnn_status != "found":
            self.device = "cpu"
            self.compute_type = "int8"
            print("CPU 모드 사용 - CUDNN 문제 또는 강제 CPU 모드 설정")
        else:
            try:
                # PyTorch CUDA 기능 확인
                x = torch.zeros(1, device="cuda")
                del x
                print("CUDA 기능 테스트 성공")

                self.device = "cuda"

                # CUDA 아키텍처에 따른 compute_type 설정
                gpu_name = torch.cuda.get_device_name(0).lower()
                if any(arch in gpu_name for arch in ['a100', 'a10', 'h100']):
                    self.compute_type = "float16"  # Ampere/Hopper 아키텍처
                    print("최신 GPU 아키텍처 감지: float16 사용")
                else:
                    self.compute_type = "int8"  # 구형 GPU
                    print("이전 GPU 아키텍처 감지: int8 사용")
            except Exception as e:
                print(f"CUDA 테스트 실패: {str(e)}")
                self.device = "cpu"
                self.compute_type = "int8"

        print(f"최종 설정 - Device: {self.device}, Compute type: {self.compute_type}")

        # 환경 변수 설정 (CUDNN 문제 해결 시도)
        if self.device == "cuda":
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 첫 번째 GPU 선택
                # CTranslate2 로깅 활성화
                os.environ["CT2_VERBOSE"] = "1"
            except Exception as e:
                print(f"환경 변수 설정 실패: {str(e)}")

        # 로컬 Faster Whisper 모델 로드
        print(f"Loading Faster Whisper {model_size} model...")
        try:
            # GPU 또는 CPU 모드에 따른 초기화
            if self.device == "cuda":
                try:
                    self.model = WhisperModel(
                        model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                        download_root="./models",
                        local_files_only=False,
                        # local_files_only=True,
                    )
                    print("Faster Whisper model loaded successfully (GPU mode)")
                except Exception as e:
                    print(f"GPU 모델 로딩 실패: {str(e)}")
                    print("compute_type 변경 시도...")

                    # compute_type 변경 시도
                    for compute_type in ["int8", "int8_float16", "float32"]:
                        try:
                            print(f"{compute_type} 시도 중...")
                            self.model = WhisperModel(
                                model_size,
                                device=self.device,
                                compute_type=compute_type,
                                download_root="./models"
                            )
                            self.compute_type = compute_type
                            print(f"Faster Whisper model loaded successfully with {compute_type}")
                            break
                        except Exception as e2:
                            print(f"{compute_type} 실패: {str(e2)}")
                    else:
                        # 모든 compute_type이 실패하면 CPU로 폴백
                        print("모든 GPU 옵션 실패, CPU 모드로 전환")
                        self.device = "cpu"
                        self.compute_type = "int8"
                        self.model = WhisperModel(
                            model_size,
                            device="cpu",
                            compute_type="int8",
                            download_root="./models",
                            cpu_threads=4
                        )
                        print("Faster Whisper model loaded successfully (CPU fallback)")
            else:
                # CPU 모드 초기화
                self.model = WhisperModel(
                    model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root="./models",
                    cpu_threads=4
                )
                print("Faster Whisper model loaded successfully (CPU mode)")
        except Exception as e:
            print(f"모델 로딩 최종 실패: {str(e)}")
            raise RuntimeError(f"Faster Whisper 모델을 로드할 수 없습니다: {str(e)}")

    def segment_audio(self, audio_file: str, start: float, end: float, output_file: str) -> str:
        """주어진 구간의 오디오를 잘라서 저장합니다."""
        y, sr = librosa.load(audio_file, sr=self.sr)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        sf.write(output_file, y[start_sample:end_sample], sr)
        return output_file, y[start_sample:end_sample], start_sample, end_sample

    def transcribe_segment(self, segment_file: str, prompt: str = "", log_prob_threshold: float = -1.0) -> str:
        """세그먼트별 음성 인식 수행 (이전 프롬프트 활용)."""
        try:
            # Faster Whisper 모델로 인식 수행
            segments, _ = self.model.transcribe(
                segment_file,
                language=self.language,
                initial_prompt=prompt,
                beam_size=5,
                length_penalty=0.99,
                repetition_penalty=1.5,
                temperature=0.,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"threshold": 0.5}
            )

            # 🎯 log_prob_threshold 적용: 신뢰도 낮은 문장 제거
            filtered_segments = [seg for seg in segments if seg.avg_logprob > log_prob_threshold]

            # 결과 텍스트 합치기
            full_text = " ".join([segment.text for segment in filtered_segments])

            # 결과 반환
            return full_text.strip()

        except Exception as e:
            print(f"음성 인식 오류: {str(e)}")
            return ""

    def transcribe_segments(self, audio_file: str, segments: List[Dict]) -> Dict[str, List[Dict]]:
        """세그먼트별 프롬프트 업데이트 방식으로 음성 인식 수행."""
        # speaker_transcripts = {"SPK0": [], "SPK1": []}
        speaker_transcripts = {}
        prompt = ""  # 프롬프트 초기화

        total_segments = len(segments)
        for idx, segment in enumerate(segments):
            print(f"처리 중: 세그먼트 {idx + 1}/{total_segments}")
            # 세그먼트 자르기
            segment_file = f"segment_{idx}.wav"
            _, y, start_sample, end_sample = self.segment_audio(audio_file, segment['start'], segment['end'],
                                                                segment_file)

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

            speaker = segment['speaker']
            if speaker not in speaker_transcripts:
                speaker_transcripts[speaker] = []

            # 결과 저장
            speaker_transcripts[segment['speaker']].append({
                "start": segment['start'],
                "end": segment['end'],
                "text": text,
                "audio": audio_binary  # 해당 구간의 오디오 데이터 저장
            })
            print(f"세그먼트 {idx + 1} 결과: {text[:50]}..." if len(text) > 50 else f"세그먼트 {idx + 1} 결과: {text}")

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

    # Hugging Face 토큰 설정 (필요시)
    hf_token = os.getenv("HF_TOKEN")

    # 오디오 파일 및 화자 구분 정보 불러오기
    audio_file = "audio.wav"
    diarizer = SpeakerDiarizer()
    diarization_results = diarizer.process_audio(audio_file)
    segments = diarizer.get_results_as_list(diarization_results)

    # GPU 사용 시도 (force_cpu=False)
    transcriber = LocalFasterWhisperTranscriber(model_size="large-v3", hf_token=hf_token, force_cpu=False)
    transcripts = transcriber.transcribe_segments(audio_file, segments)

    # 결과 출력
    print("\n=== 트랜스크립션 결과 ===\n")
    print(transcriber.format_conversation(transcripts))