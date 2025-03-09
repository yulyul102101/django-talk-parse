# 이미 시스템에 있는 PyTorch 이미지 사용 (CUDNN 9 포함)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    git \
    wget && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

# CUDNN 심볼릭 링크 생성
RUN echo "CUDNN 설정 시작" && \
    mkdir -p /usr/lib/x86_64-linux-gnu/ && \
    find /opt -name "libcudnn*.so*" -exec ln -sf {} /usr/lib/x86_64-linux-gnu/ \; && \
    # CUDNN 8 호환성을 위한 추가 링크
    ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9 /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8 && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.9 /usr/lib/x86_64-linux-gnu/libcudnn.so.8 && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9 /usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8 && \
    echo 'export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/ctranslate2.libs:$LD_LIBRARY_PATH' > /etc/profile.d/cudnn_path.sh && \
    chmod +x /etc/profile.d/cudnn_path.sh

# 작업 디렉토리 설정
WORKDIR /data/django-talk-parse

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY .. .

# LD_LIBRARY_PATH 설정
RUN echo 'export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/ctranslate2.libs:$LD_LIBRARY_PATH' >> /root/.bashrc

# 엔트리포인트 설정
RUN echo '#!/bin/bash\nexport LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/ctranslate2.libs:$LD_LIBRARY_PATH\nexec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]
