# GPT Killer - AI 텍스트 분석기

GPT Killer는 텍스트가 AI에 의해 생성되었는지 여부와 표절 여부를 분석하는 웹 애플리케이션입니다.

## 주요 기능

- GPT 생성 확률 분석
  - 문장 길이 분석 (25단어 이상)
  - 단어 반복 패턴 분석 (4회 이상)
  - 특정 문장 패턴 분석 (접속사, 동사, 형용사, 인용, '것' 패턴)
- 표절 검사
  - 8어절 이상의 문장 대상
  - 인터넷 자료와 비교
  - 표절률 계산

## 기술 스택

- Backend: Python, Flask
- Frontend: HTML, CSS, JavaScript
- 배포: Render

## 설치 및 실행

1. 저장소 클론
```bash
git clone https://github.com/yourusername/gpt-killer.git
cd gpt-killer
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. 서버 실행
```bash
python app.py
```

## 라이선스

MIT License 