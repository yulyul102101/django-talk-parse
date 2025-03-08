from typing import Dict
import requests
import json
from dotenv import load_dotenv
import os

class VoicePhishingDetector:
    def __init__(self, model_name="deepseek-r1:32b"):
        load_dotenv()
        self.model_name = model_name
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
    def analyze_conversation(self, transcript: str) -> Dict[str, str]:
        """
        텍스트로 변환된 대화 내용을 분석하여 보이스피싱 여부를 판단합니다.
        
        Args:
            transcript (str): 텍스트로 변환된 대화 내용
            
        Returns:
            Dict[str, str]: 판단 결과와 근거를 포함한 사전
        """
        prompt = self._create_prompt(transcript)
        response = self._get_ollama_response(prompt)
        return self._parse_response(response)

    def _create_prompt(self, transcript: str) -> str:
        """
        LLM에 전달할 프롬프트를 생성합니다.
        
        Args:
            transcript (str): 텍스트로 변환된 대화 내용
            
        Returns:
            str: 생성된 프롬프트
        """
        return f"""
        다음은 전화 통화의 녹음 내용을 텍스트로 변환한 것입니다. 이 대화가 보이스피싱인지 분석해주세요.

        통화 내용:
        {transcript}

        위 대화를 분석하여 다음 형식으로 답변해주세요:
        1. 판단: [보이스피싱 / 정상 통화 / 판단 어려움] 중 하나를 선택
        2. 근거: 판단의 근거를 상세히 설명해주세요. 가능하다면 대화 참여자를 구분하여 설명해주세요.
        """

    def _get_ollama_response(self, prompt: str) -> str:
        """
        Ollama API를 호출하여 응답을 얻습니다.
        
        Args:
            prompt (str): LLM에 전달할 프롬프트
            
        Returns:
            str: LLM의 응답
        """
        url = f"{self.ollama_base_url}/api/generate"
        
        data = {
            "model": self.model_name,
            "prompt": "당신은 보이스피싱을 탐지하는 전문가입니다.\n\n" + prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Ollama API 호출 중 오류가 발생했습니다: {e}")
            return f"오류: {str(e)}"

    def _parse_response(self, response: str) -> Dict[str, str]:
        """
        LLM의 응답을 파싱하여 판단 결과와 근거를 추출합니다.
        
        Args:
            response (str): LLM의 응답
            
        Returns:
            Dict[str, str]: 판단 결과와 근거를 포함한 사전
        """
        lines = response.split('\n')
        result = {}
        current_key = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("1. 판단:"):
                result["judgment"] = line.split(":", 1)[1].strip()
            elif line.startswith("2. 근거:"):
                current_key = "evidence"
                result[current_key] = line.split(":", 1)[1].strip()
            elif current_key == "evidence":
                # 증거 부분의 추가 텍스트를 연결
                result[current_key] += "\n" + line
        
        # 기본값 설정
        if "judgment" not in result:
            result["judgment"] = "판단 어려움"
        if "evidence" not in result:
            result["evidence"] = "모델 응답에서 근거를 추출할 수 없습니다."
            
        return result


# 사용 예시
if __name__ == "__main__":
    detector = VoicePhishingDetector()
    
    # my_whisper.py에서 생성된 텍스트 파일을 읽어옵니다.
    with open("audio_transcript.txt", "r", encoding="utf-8") as f:
        transcript = f.read()
    
    result = detector.analyze_conversation(transcript)
    print("판단:", result["judgment"])
    print("근거:", result["evidence"])    
