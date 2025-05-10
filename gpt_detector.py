import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import logging

logger = logging.getLogger(__name__)

class GPTDetector:
    def __init__(self):
        logger.debug("GPT Detector 초기화 시작")
        try:
            self.model = AutoModelForCausalLM.from_pretrained('skt/kogpt2-base-v2')
            self.tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            logger.debug("GPT Detector 초기화 완료")
        except Exception as e:
            logger.error(f"GPT Detector 초기화 중 오류 발생: {str(e)}")
            raise

    def calculate_perplexity(self, text):
        try:
            encodings = self.tokenizer(text, return_tensors='pt')
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            
            with torch.no_grad():
                outputs = self.model(**encodings, labels=encodings['input_ids'])
                loss = outputs.loss
            
            perplexity = torch.exp(loss).item()
            logger.debug(f"Perplexity 계산 완료: {perplexity}")
            return perplexity
        except Exception as e:
            logger.error(f"Perplexity 계산 중 오류 발생: {str(e)}")
            raise

    def calculate_burstiness(self, text):
        try:
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return 0
            
            lengths = [len(s) for s in sentences]
            mean = np.mean(lengths)
            std = np.std(lengths)
            
            burstiness = std / mean if mean > 0 else 0
            logger.debug(f"Burstiness 계산 완료: {burstiness}")
            return burstiness
        except Exception as e:
            logger.error(f"Burstiness 계산 중 오류 발생: {str(e)}")
            raise

    def analyze_text(self, text):
        try:
            logger.debug("텍스트 분석 시작")
            perplexity = self.calculate_perplexity(text)
            burstiness = self.calculate_burstiness(text)
            
            logger.debug(f"Perplexity 값: {perplexity:.2f}")
            logger.debug(f"Burstiness 값: {burstiness:.2f}")
            
            # 판단 기준
            if perplexity < 30 and burstiness < 8:
                result = "GPT 의심 (AI 가능성 높음)"
            elif perplexity < 50:
                result = "중립 (혼합 가능성)"
            else:
                result = "인간 작성 가능성 높음"
            
            logger.debug(f"판단 결과: {result}")
            
            return {
                'perplexity': perplexity,
                'burstiness': burstiness,
                'result': result
            }
        except Exception as e:
            logger.error(f"텍스트 분석 중 오류 발생: {str(e)}")
            raise

def main():
    # 테스트
    detector = GPTDetector()
    
    # 테스트 텍스트
    text = """
    현대 사회에서 정보는 매우 중요한 자산이다. 다양한 정보들이 인터넷을 통해 전송되고 있으며, 
    이러한 정보들이 안전하게 보호되기 위해서는 강력한 보안 기술이 필요하다. 
    현재까지 우리는 주로 공개키 암호화 방식을 사용해왔지만, 양자 컴퓨터의 등장으로 
    이러한 방식의 안전성이 위협받고 있다. 따라서 새로운 암호화 방식의 개발이 시급한 상황이다.
    """
    
    # 분석 실행
    detector.analyze_text(text)

if __name__ == "__main__":
    main() 