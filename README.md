$ cd /data

$ git clone https://github.com/yulyul1021/django-talk-parse.git

$ cd /data/django-talk-parse

$ vim .env

    HF_TOKEN=    
    
    OLLAMA_BASE_URL=http://localhost:11434

$ cd /data/django-talk-parse/docker

$ docker-compose up --build -d

$ docker exec -it call_analysis /bin/bash

$ ollama serve

$ ollama run deepseek-r1:32b # ollama API 서버(로컬) 실행

// >>> # 에서 Ctrl+D(종료)


$ python manage.py makemigrations

$ python manage.py migrate --run-syncdb 

$ python manage.py runserver 0.0.0.0:17777 # Django 서버 실행


// 컨테이너 종료 및 이미지 삭제

$ cd /data/django-talk-parse/docker

$ docker-compose down

$ docker rmi django_call_analysis:latest