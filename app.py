import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from gpt_killer import GPTKiller
import docx
import logging
from gpt_detector import GPTDetector

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 업로드 폴더가 없으면 생성
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

ALLOWED_EXTENSIONS = {'txt', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_docx(file_path):
    """DOCX 파일에서 텍스트를 추출합니다."""
    try:
        doc = docx.Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        logger.error(f"DOCX 파일 처리 중 오류: {str(e)}")
        return None

detector = GPTDetector()

@app.route('/')
def index():
    logger.debug("메인 페이지 요청")
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        logger.debug("분석 요청 받음")
        logger.debug(f"요청 데이터: {request.form}")
        logger.debug(f"요청 파일: {request.files}")
        
        if 'file' in request.files:
            logger.debug("파일 업로드 요청 감지")
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                logger.debug(f"파일 저장됨: {file_path}")
                
                try:
                    if filename.endswith('.docx'):
                        text = extract_text_from_docx(file_path)
                    else:  # .txt
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    
                    if text is None:
                        logger.error("파일 처리 실패")
                        return jsonify({'error': '파일을 처리할 수 없습니다.'}), 400
                    
                    logger.debug(f"파일에서 추출된 텍스트: {text[:100]}...")
                    
                finally:
                    # 파일 처리 후 삭제
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.debug(f"임시 파일 삭제됨: {file_path}")
            else:
                logger.error(f"허용되지 않는 파일 형식: {file.filename if file else 'None'}")
                return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400
        else:
            logger.debug("텍스트 입력 요청 감지")
            text = request.form.get('text', '')
            logger.debug(f"받은 텍스트: {text[:100]}...")  # 처음 100자만 로깅
            
            if not text:
                logger.error("텍스트가 비어있음")
                return jsonify({'error': '텍스트를 입력해주세요.'}), 400
        
        logger.debug("GPT Detector 분석 시작")
        try:
            result = detector.analyze_text(text)
            logger.debug(f"분석 결과: {result}")
            return jsonify(result)
        except Exception as e:
            logger.error(f"GPT Detector 분석 중 오류 발생: {str(e)}", exc_info=True)
            return jsonify({'error': '텍스트 분석 중 오류가 발생했습니다.'}), 500
        
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({'error': '서버 오류가 발생했습니다.'}), 500

@app.route('/result', methods=['POST'])
def result():
    text = request.form.get('text', '')
    if not text:
        return redirect(url_for('index'))
    
    try:
        # 텍스트 분석
        result = detector.analyze_text(text)
        
        # 결과 템플릿에 전달
        return render_template('result.html',
                             text=text,
                             perplexity=result['perplexity'],
                             burstiness=result['burstiness'],
                             judgement=result['result'])
    except Exception as e:
        app.logger.error(f"분석 중 오류 발생: {str(e)}")
        return render_template('index.html', error="분석 중 오류가 발생했습니다.")

if __name__ == '__main__':
    app.run(debug=True, port=5001) 