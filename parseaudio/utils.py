import subprocess


def get_installed_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            if len(lines) > 1:  # 헤더 제외 후 모델명 추출
                models = [line.split()[0] for line in lines[1:] if line.strip()]
                return models if models else []
        else:
            print(f"Error: ollama list 실행 실패, 코드 {result.returncode}")
            print(f"출력: {result.stdout}")
            print(f"오류: {result.stderr}")
        return []
    except Exception as e:
        print(f"예외 발생: {e}")
        return []


def stop_running_model(model_name):
    """
    실행 중인 Ollama 모델을 확인하고, 선택한 모델이 실행 중이라면 종료한다.
    """
    try:
        # 실행 중인 모델 확인 (ollama ps)
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, shell=True)

        if result.returncode == 0:
            output = result.stdout.strip()
            print(f"실행 중인 모델 확인:\n{output}")  # 디버깅용

            lines = output.splitlines()
            if len(lines) > 1:  # 첫 번째 줄은 헤더이므로 제외
                running_models = [line.split()[0] for line in lines[1:] if line.strip()]

                if model_name in running_models:
                    print(f"모델 '{model_name}' 실행 중 → 종료 시도")
                    stop_result = subprocess.run(["ollama", "stop", model_name], capture_output=True, text=True,
                                                 shell=True)

                    if stop_result.returncode == 0:
                        print(f"모델 '{model_name}' 종료 완료")
                    else:
                        print(f"모델 '{model_name}' 종료 실패: {stop_result.stderr}")
                else:
                    print(f"모델 '{model_name}'은 실행 중이 아님.")
            else:
                print("현재 실행 중인 Ollama 모델 없음.")
        else:
            print(f"ollama ps 실행 실패, 코드 {result.returncode}, 오류: {result.stderr}")

    except Exception as e:
        print(f"예외 발생: {e}")