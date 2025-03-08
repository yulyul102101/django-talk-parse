from pyannote.audio import Pipeline
from dotenv import load_dotenv
import torch
import os
from typing import List, Tuple

class SpeakerDiarizer:
    def __init__(self):
        # .env 파일 로드
        load_dotenv()
        
        # 환경 변수에서 HF_TOKEN 가져오기
        self.hf_token = os.getenv('HF_TOKEN')
        if not self.hf_token:
            raise ValueError("HF_TOKEN이 설정되지 않았습니다. .env 파일을 확인해주세요.")
        
        # 파이프라인 초기화
        self.pipeline = self._initialize_pipeline()

    def _initialize_pipeline(self) -> Pipeline:
        # 파이프라인 로딩
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token
        )
        
        # GPU 사용 가능시 GPU로 이동
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        
        return pipeline

    def process_audio(self, audio_path: str) -> List[Tuple[float, float, str]]:
        """
        오디오 파일을 처리하여 화자 분리 결과를 반환합니다.
        화자는 항상 2명(SPEAKER_00, SPEAKER_01)으로 설정됩니다.
        
        Args:
            audio_path (str): 처리할 오디오 파일 경로
            
        Returns:
            List[Tuple[float, float, str]]: (시작 시간, 종료 시간, 화자) 튜플의 리스트
        """
        try:
            # 화자 분리 수행 (num_speakers=2로 고정)
            diarization_result = self.pipeline(audio_path, num_speakers=2)
            
            # 결과를 리스트로 변환
            segments = []
            for segment, _, speaker in diarization_result.itertracks(yield_label=True):
                # SPEAKER_00을 '발신자', SPEAKER_01을 '수신자'로 변경
                #speaker_label = "발신자" if speaker == "SPEAKER_00" else "수신자"
                #segments.append((segment.start, segment.end, speaker_label))
                
                speaker_label = "SPK0" if speaker == "SPEAKER_00" else "SPK1"
                segments.append((segment.start, segment.end, speaker_label))
            return segments
            
        except Exception as e:
            print(f"화자 분리 중 오류 발생: {str(e)}")
            return []

    def get_results_as_list(self, segments: List[Tuple[float, float, str]]) -> List[dict]:
        """
        화자 분리 결과를 리스트 형태로 반환합니다.
        
        Returns:
            List[dict]: 각 세그먼트의 정보를 담은 딕셔너리 리스트
        """
        return [
            {
                "speaker": speaker,
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2)
            }
            for start, end, speaker in segments
        ]

    def get_results_as_dict(self, segments: List[Tuple[float, float, str]]) -> dict:
        """
        화자 분리 결과를 화자별로 그룹화하여 딕셔너리 형태로 반환합니다.
        
        Returns:
            dict: 화자별 발화 정보를 담은 딕셔너리
        """
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

    # print_results 메서드는 필요한 경우 아래와 같이 수정하여 사용
    def print_results(self, segments: List[Tuple[float, float, str]]) -> None:
        """화자 분리 결과를 콘솔에 출력합니다."""
        for start, end, speaker in segments:
            print(f"화자: {speaker}\t시작: {start:.2f}s\t종료: {end:.2f}s")

# 사용 예시
if __name__ == "__main__":
    diarizer = SpeakerDiarizer()
    
    # 오디오 파일 경로 지정
    audio_file = "audio.wav"
    
    # 화자 분리 수행
    results = diarizer.process_audio(audio_file)
    
    # 결과 출력
    if results:
        print("화자 분리 결과:")
        diarizer.print_results(results)
    else:
        print("화자 분리에 실패했습니다.")