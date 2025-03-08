# 베이스 이미지 설정
FROM python:3.10

# 작업 디렉토리 설정
WORKDIR /data

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY . /data/

# 포트 노출
EXPOSE 17777

# 컨테이너 유지 (tail -f /dev/null)
ENTRYPOINT ["tail", "-f", "/dev/null"]
