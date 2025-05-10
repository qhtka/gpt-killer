# GPT 텍스트 탐지기

이 프로젝트는 텍스트가 GPT와 같은 AI에 의해 생성되었는지, 아니면 인간이 작성했는지 분석하는 웹 애플리케이션입니다.

## 주요 기능

- Perplexity와 Burstiness 기반의 텍스트 분석
- GPT 생성 가능성 판단
- 시각적 분석 결과 제공

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/gpt-detector.git
cd gpt-detector
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

3. 서버 실행
```bash
python app.py
```

## 사용 방법

1. 웹 브라우저에서 `http://localhost:5001` 접속
2. 분석하고 싶은 텍스트 입력
3. 분석 결과 확인

## 분석 기준

- **Perplexity**
  - GPT 의심: < 30
  - 중립: 30 ~ 50
  - 인간 작성: > 50

- **Burstiness**
  - GPT 의심: < 8
  - 중립: 8 ~ 15
  - 인간 작성: > 15

## 기술 스택

- Python 3.11
- Flask
- PyTorch
- Transformers
- Chart.js 