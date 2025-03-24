from pyannote.audio import Pipeline
from dotenv import load_dotenv
import torch
import os
import numpy as np
import soundfile as sf
import librosa
from typing import List, Tuple, Dict
import time  # 시간 측정 추가


class SpeakerDiarizer:
    def __init__(self, num_speakers=3):
        # 초기화 시간 측정 시작
        self.start_time = time.time()
        print(f"SpeakerDiarizer 초기화 시작: {self.start_time}")

        # 화자 수 설정 (기본값: 3)
        self.num_speakers = num_speakers
        print(f"화자 수: {self.num_speakers}명")

        # GPU 사용 가능 여부 먼저 확인
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU 감지됨: {gpu_name}")
            # GPU 메모리 정보 출력
            if hasattr(torch.cuda, 'memory_allocated'):
                print(f"현재 GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
                print(f"최대 GPU 메모리 할당량: {torch.cuda.max_memory_allocated(0) / 1024 ** 2:.2f} MB")
        else:
            print("GPU를 찾을 수 없습니다. CPU를 사용합니다.")

        # .env 파일 로드
        load_dotenv()

        # 환경 변수에서 HF_TOKEN 가져오기
        self.hf_token = os.getenv('HF_TOKEN')
        if not self.hf_token:
            raise ValueError("HF_TOKEN이 설정되지 않았습니다. .env 파일을 확인해주세요.")

        # 파이프라인 초기화
        self.pipeline = self._initialize_pipeline()

        # 표준 샘플링 레이트
        self.sr = 16000

        # 초기화 시간 측정 완료
        self.init_time = time.time() - self.start_time
        print(f"SpeakerDiarizer 초기화 완료: {self.init_time:.4f}초 소요")

    def _initialize_pipeline(self) -> Pipeline:
        print("파이프라인 로드 시작...")
        pipeline_start = time.time()

        # 파이프라인 로딩 - 명시적으로 device 지정
        if self.has_gpu:
            # GPU 설정을 명시적으로 지정
            device = torch.device("cuda:0")
            # CUDA 장치로 설정된 파이프라인 생성
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            ).to(device)

            # 모델 파라미터가 실제로 CUDA에 있는지 확인
            on_cuda = self._verify_cuda_model(pipeline)

            if on_cuda:
                print("✅ Speaker Diarization: GPU에 성공적으로 로드됨")
            else:
                print("⚠️ Speaker Diarization: to(cuda) 호출했으나 모델이 CPU에 있음")
                print("    -> 강제로 GPU로 이동 시도")
                # 파이프라인 내의 모델들을 명시적으로 GPU로 이동
                self._force_to_cuda(pipeline)
        else:
            # CPU로 설정
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            print("Speaker Diarization: CPU 사용 중")

        pipeline_time = time.time() - pipeline_start
        print(f"파이프라인 로드 시간: {pipeline_time:.4f}초")

        return pipeline

    def _verify_cuda_model(self, pipeline) -> bool:
        """파이프라인 내 모델들이 실제로 CUDA에 있는지 확인"""
        if hasattr(pipeline, "model"):
            # 파이프라인의 모델 매개변수가 CUDA에 있는지 확인
            try:
                return next(pipeline.model.parameters()).is_cuda
            except (AttributeError, StopIteration):
                pass

        # 더 깊게 들어가서 모든 하위 모델/속성 확인
        return self._search_cuda_tensors(pipeline)

    def _search_cuda_tensors(self, obj) -> bool:
        """객체 내에서 CUDA 텐서를 재귀적으로 검색"""
        if hasattr(obj, "parameters"):
            try:
                for param in obj.parameters():
                    if param.is_cuda:
                        return True
            except (AttributeError, TypeError):
                pass

        # 하위 속성 검색
        for attr_name in dir(obj):
            # 스페셜 메소드나 내장 속성 건너뛰기
            if attr_name.startswith('__') or attr_name in ['to', 'cpu', 'cuda']:
                continue

            try:
                attr = getattr(obj, attr_name)
                # 텐서인 경우 CUDA 확인
                if isinstance(attr, torch.Tensor) and attr.is_cuda:
                    return True
                # 다른 객체인 경우 재귀적으로 검색
                elif hasattr(attr, '__dict__') and not callable(attr):
                    if self._search_cuda_tensors(attr):
                        return True
            except (AttributeError, RuntimeError):
                continue

        return False

    def _force_to_cuda(self, pipeline):
        """파이프라인의 모든 모델을 명시적으로 CUDA로 이동 시도"""
        # 파이프라인 내의 주요 모델 속성들을 CUDA로 이동
        for attr_name in dir(pipeline):
            if attr_name.startswith('__'):
                continue

            try:
                attr = getattr(pipeline, attr_name)
                if hasattr(attr, 'to') and callable(attr.to):
                    setattr(pipeline, attr_name, attr.to('cuda'))
                    print(f"  {attr_name} -> CUDA로 이동됨")
            except (AttributeError, RuntimeError, TypeError):
                continue

        # 가능한 모든 모델 구성요소들 찾아서 이동
        if hasattr(pipeline, 'model') and hasattr(pipeline.model, 'to'):
            pipeline.model = pipeline.model.to('cuda')
            print("  pipeline.model -> CUDA로 이동됨")

        if hasattr(pipeline, 'segmentation') and hasattr(pipeline.segmentation, 'to'):
            pipeline.segmentation = pipeline.segmentation.to('cuda')
            print("  pipeline.segmentation -> CUDA로 이동됨")

        if hasattr(pipeline, 'embedding') and hasattr(pipeline.embedding, 'to'):
            pipeline.embedding = pipeline.embedding.to('cuda')
            print("  pipeline.embedding -> CUDA로 이동됨")

        # 확인
        print("GPU 메모리 사용량 (강제 이동 후):",
              f"{torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")

    def process_audio(self, audio_path: str, num_speakers=None) -> List[Tuple[float, float, str]]:
        """
        오디오 파일을 처리하여 화자 분리 결과를 반환합니다.
        화자 수는 생성자에서 설정한 값이나 함수 호출 시 지정한 값을 사용합니다.
        """
        process_start = time.time()
        print(f"오디오 처리 시작: {process_start}")

        # 화자 수 결정 (함수 호출 시 지정한 값이 있으면 우선 사용)
        speakers_count = num_speakers if num_speakers is not None else self.num_speakers
        print(f"화자 분리 실행: {speakers_count}명으로 설정")

        try:
            # GPU 메모리 사용량 (처리 전)
            if self.has_gpu:
                print(f"처리 전 GPU 메모리: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")

            # 화자 분리 수행 (화자 수 지정)
            diarization_result = self.pipeline(audio_path, num_speakers=speakers_count)

            # GPU 메모리 사용량 (처리 후)
            if self.has_gpu:
                print(f"처리 후 GPU 메모리: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")

            # 결과를 리스트로 변환
            segments = []
            for segment, _, speaker in diarization_result.itertracks(yield_label=True):
                # 화자 라벨을 SPK0, SPK1, SPK2 형태로 변환
                speaker_num = int(speaker.split('_')[1])
                speaker_label = f"SPK{speaker_num}"
                segments.append((segment.start, segment.end, speaker_label))

            process_time = time.time() - process_start
            print(f"오디오 처리 완료: {process_time:.4f}초 소요 (세그먼트 수: {len(segments)})")
            return segments

        except Exception as e:
            process_time = time.time() - process_start
            print(f"화자 분리 중 오류 발생: {str(e)}")
            print(f"에러 발생까지 소요 시간: {process_time:.4f}초")
            return []

    def get_results_as_list(self, segments: List[Tuple[float, float, str]]) -> List[Dict]:
        """화자 분리 결과를 리스트 형태로 반환합니다."""
        return [
            {
                "speaker": speaker,
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2)
            }
            for start, end, speaker in segments
        ]

    def get_results_as_dict(self, segments: List[Tuple[float, float, str]]) -> Dict:
        """화자 분리 결과를 화자별로 그룹화하여 딕셔너리 형태로 반환합니다."""
        speaker_segments = {}
        for start, end, speaker in segments:
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []

            speaker_segments[speaker].append({
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2)
            })

        return speaker_segments

    def create_padded_segments(self, audio_path: str, segments: List[Dict], pad_seconds: float = 4.0,
                               output_dir: str = "padded_segments") -> List[Dict]:
        """세그먼트를 추출하고 앞뒤로 제로 패딩(무음)을 적용하여 저장합니다."""
        padding_start = time.time()
        print(f"세그먼트 패딩 시작: {padding_start}")

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 원본 오디오 로드
        y, sr = librosa.load(audio_path, sr=self.sr)

        padded_segments = []

        for i, segment in enumerate(segments):
            # 세그먼트 시작/종료 시간
            start_sample = int(segment["start"] * sr)
            end_sample = int(segment["end"] * sr)

            # 패딩 샘플 수 계산
            pad_samples = int(pad_seconds * sr)

            # 세그먼트 오디오 데이터 추출
            segment_audio = y[start_sample:end_sample]

            # 앞뒤로 제로 패딩 적용
            padded_audio = np.pad(segment_audio, (pad_samples, pad_samples), 'constant', constant_values=0)

            # 패딩된 세그먼트 저장
            output_file = os.path.join(output_dir, f"segment_{i:04d}_{segment['speaker']}.wav")
            sf.write(output_file, padded_audio, sr)

            # 결과 정보에 추가
            padded_segments.append({
                "speaker": segment["speaker"],
                "original_start": segment["start"],
                "original_end": segment["end"],
                "padded_start": segment["start"],  # 원래 시작 시간 유지 (제로패딩은 물리적으로 추가됨)
                "padded_end": segment["end"],  # 원래 종료 시간 유지 (제로패딩은 물리적으로 추가됨)
                "duration": (end_sample - start_sample) / sr + pad_seconds * 2,  # 패딩 포함 총 길이
                "file_path": output_file
            })

        padding_time = time.time() - padding_start
        print(f"{len(padded_segments)}개의 제로 패딩된 세그먼트가 {output_dir}에 저장되었습니다. (소요 시간: {padding_time:.4f}초)")
        return padded_segments

    def print_results(self, segments: List[Tuple[float, float, str]]) -> None:
        """화자 분리 결과를 콘솔에 출력합니다."""
        # 화자별 발화 시간 계산
        speaker_durations = {}
        for start, end, speaker in segments:
            if speaker not in speaker_durations:
                speaker_durations[speaker] = 0
            speaker_durations[speaker] += (end - start)

        # 화자별 발화 시간 출력
        print("\n===== 화자별 발화 시간 =====")
        for speaker, duration in speaker_durations.items():
            print(f"{speaker}: {duration:.2f}초")

        # 세그먼트 정보 출력
        print("\n===== 세그먼트 정보 =====")
        for start, end, speaker in segments:
            print(f"화자: {speaker}\t시작: {start:.2f}s\t종료: {end:.2f}s\t길이: {end - start:.2f}s")


# 사용 예시
if __name__ == "__main__":
    total_start_time = time.time()
    print(f"전체 프로그램 시작 시간: {total_start_time}")

    # SpeakerDiarizer 초기화 (화자 수 3명으로 설정)
    diarizer_start_time = time.time()
    print(f"SpeakerDiarizer 초기화 시작: {diarizer_start_time}")

    diarizer = SpeakerDiarizer(num_speakers=3)

    diarizer_init_time = time.time() - diarizer_start_time
    print(f"SpeakerDiarizer 초기화 완료: {diarizer_init_time:.4f}초 소요")

    # 오디오 파일 경로 지정
    audio_file = r"E:\hello_E\django-talk-parse-main\django-talk-parse-main\parseaudio\parsing\audio.wav"

    # 화자 분리 수행
    process_start_time = time.time()
    print(f"화자 분리 처리 시작: {process_start_time}")

    results = diarizer.process_audio(audio_file)

    process_time = time.time() - process_start_time
    print(f"화자 분리 처리 완료: {process_time:.4f}초 소요")

    # 결과를 리스트로 변환
    segments_list = diarizer.get_results_as_list(results)

    # 패딩된 세그먼트 생성 (앞뒤로 4초씩 제로패딩)
    padding_start_time = time.time()
    print(f"세그먼트 패딩 시작: {padding_start_time}")

    padded_segments = diarizer.create_padded_segments(audio_file, segments_list, pad_seconds=4.0)

    padding_time = time.time() - padding_start_time
    print(f"세그먼트 패딩 완료: {padding_time:.4f}초 소요")

    # 결과 출력
    if results:
        print("\n화자 분리 결과:")
        diarizer.print_results(results)
    else:
        print("화자 분리에 실패했습니다.")

    total_time = time.time() - total_start_time
    print(f"\n전체 프로그램 실행 시간: {total_time:.4f}초")

    # 단계별 실행 시간 요약
    print("\n===== 실행 시간 요약 =====")
    print(f"1. SpeakerDiarizer 초기화: {diarizer.init_time:.4f}초")
    print(f"2. 화자 분리 처리: {process_time:.4f}초")
    print(f"3. 세그먼트 패딩: {padding_time:.4f}초")
    print(f"총 실행 시간: {total_time:.4f}초")