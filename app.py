import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from gpt_killer import GPTKiller
import docx
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        gpt_killer = GPTKiller()
        
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                try:
                    if filename.endswith('.docx'):
                        text = extract_text_from_docx(file_path)
                    else:  # .txt
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    
                    if text is None:
                        return jsonify({'error': '파일을 처리할 수 없습니다.'}), 400
                    
                finally:
                    # 파일 처리 후 삭제
                    if os.path.exists(file_path):
                        os.remove(file_path)
            else:
                return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400
        else:
            text = request.form.get('text', '')
            if not text:
                return jsonify({'error': '텍스트를 입력해주세요.'}), 400
        
        result = gpt_killer.analyze_text(text)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {str(e)}")
        return jsonify({'error': '서버 오류가 발생했습니다.'}), 500

if __name__ == '__main__':
    app.run(debug=True) 