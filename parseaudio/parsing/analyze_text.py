import openai
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import os

class VoicePhishingDetector:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        openai.api_key = self.api_key

    def analyze_conversation(self, transcript: str) -> Dict[str, str]:
        prompt = self._create_prompt(transcript)
        response = self._get_gpt_response(prompt)
        return self._parse_response(response)

    def _create_prompt(self, transcript: str) -> str:
        return f"""
        다음은 전화 통화의 녹음 내용을 텍스트로 변환한 것입니다. 이 대화가 보이스피싱인지 분석해주세요.

        통화 내용:
        {transcript}

        위 대화를 분석하여 다음 형식으로 답변해주세요:
        1. 판단: [보이스피싱 / 정상 통화 / 판단 어려움] 중 하나를 선택
        2. 근거: 판단의 근거를 상세히 설명해주세요. 가능하다면 대화 참여자를 구분하여 설명해주세요.
        """

    def _get_gpt_response(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 보이스피싱을 탐지하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def _parse_response(self, response: str) -> Dict[str, str]:
        lines = response.split('\n')
        result = {}
        current_key = ""
        for line in lines:
            if line.startswith("1. 판단:"):
                result["judgment"] = line.split(":")[1].strip()
            elif line.startswith("2. 근거:"):
                current_key = "evidence"
                result[current_key] = line.split(":")[1].strip()
            #elif current_key:
                #result[current_key] += line + "\n"
        return result

# 사용 예시
if __name__ == "__main__":
    detector = VoicePhishingDetector()
    
    # my_whisper.py에서 생성된 텍스트 파일을 읽어옵니다.
    with open("transcribed_call.txt", "r", encoding="utf-8") as f:
        transcript = f.read()
    
    result = detector.analyze_conversation(transcript)
    print("판단:", result["judgment"])
    print("근거:", result["evidence"])    
