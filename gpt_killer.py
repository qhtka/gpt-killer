import re
from collections import Counter
import logging
import requests
from bs4 import BeautifulSoup
import time

logger = logging.getLogger(__name__)

class GPTKiller:
    def __init__(self):
        self.search_cache = {}  # 검색 결과 캐시
        
    def analyze_text(self, text: str) -> dict:
        """
        주어진 텍스트를 분석하여 GPT 생성 여부와 표절률을 반환합니다.
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            Dict[str, float]: 분석 결과 (GPT 생성 확률, 표절률)
        """
        try:
            # 문장별 분석
            sentences = self._split_into_sentences(text)
            sentence_analyses = []
            
            for sentence in sentences:
                analysis = self._analyze_sentence_pattern(sentence)
                sentence_analyses.append({
                    'sentence': sentence,
                    'score': analysis['score'],
                    'reasons': analysis['reasons']
                })
            
            # 평균 확률 계산 (더 엄격한 기준 적용)
            scores = [analysis['score'] for analysis in sentence_analyses]
            if not scores:
                avg_score = 0.5
            else:
                # 상위 30% 문장들의 평균을 사용
                scores.sort(reverse=True)
                top_scores = scores[:max(1, int(len(scores) * 0.3))]
                avg_score = sum(top_scores) / len(top_scores)
            
            # 표절 검사
            plagiarism_results = self._check_plagiarism(text)
            
            return {
                "gpt_probability": avg_score,
                "plagiarism_rate": plagiarism_results['rate'],
                "plagiarism_details": plagiarism_results['details'],
                "sentence_analyses": sentence_analyses
            }
        except Exception as e:
            logger.error(f"텍스트 분석 중 오류: {str(e)}")
            return {
                "gpt_probability": 0.5,
                "plagiarism_rate": 0.0,
                "plagiarism_details": [],
                "sentence_analyses": []
            }
    
    def _check_plagiarism(self, text: str) -> dict:
        """텍스트의 표절 여부를 검사합니다."""
        try:
            sentences = self._split_into_sentences(text)
            plagiarism_details = []
            total_sentences = len(sentences)
            plagiarized_sentences = 0
            
            for sentence in sentences:
                # 8어절 이상인 문장만 검사
                eojeols = self._split_into_eojeols(sentence)
                if len(eojeols) >= 8:
                    # 문장 검색
                    search_results = self._search_sentence(sentence)
                    if search_results:
                        plagiarized_sentences += 1
                        plagiarism_details.append({
                            'sentence': sentence,
                            'matches': search_results
                        })
            
            # 표절률 계산
            plagiarism_rate = plagiarized_sentences / total_sentences if total_sentences > 0 else 0.0
            
            return {
                'rate': plagiarism_rate,
                'details': plagiarism_details
            }
        except Exception as e:
            logger.error(f"표절 검사 중 오류: {str(e)}")
            return {
                'rate': 0.0,
                'details': []
            }
    
    def _split_into_eojeols(self, text: str) -> list:
        """텍스트를 어절 단위로 분리합니다."""
        return text.split()
    
    def _search_sentence(self, sentence: str) -> list:
        """문장을 검색하여 일치하는 결과를 반환합니다."""
        try:
            # 캐시된 결과가 있으면 반환
            if sentence in self.search_cache:
                return self.search_cache[sentence]
            
            # 검색 API 호출 (예: Google Custom Search API)
            # 실제 구현에서는 적절한 검색 API를 사용해야 합니다
            # 여기서는 예시로 빈 결과를 반환합니다
            results = []
            
            # 캐시에 저장
            self.search_cache[sentence] = results
            return results
        except Exception as e:
            logger.error(f"문장 검색 중 오류: {str(e)}")
            return []
    
    def _analyze_sentence_pattern(self, sentence: str) -> dict:
        """문장의 패턴을 분석하여 점수와 이유를 반환합니다."""
        try:
            score = 0.0
            reasons = []
            
            # 문장 길이 분석 (더 엄격한 기준)
            words = sentence.split()
            if len(words) > 25:  # 20 -> 25
                score += 0.25  # 0.2 -> 0.25
                reasons.append({
                    'type': 'long_sentence',
                    'text': sentence,
                    'description': '긴 문장 (25단어 이상)'
                })
            
            # 반복되는 단어 패턴 (더 엄격한 기준)
            word_counts = Counter(words)
            repeated_words = [word for word, count in word_counts.items() if count > 3]  # 2 -> 3
            if repeated_words:
                score += 0.35  # 0.3 -> 0.35
                reasons.append({
                    'type': 'repeated_words',
                    'text': ', '.join(repeated_words),
                    'description': '반복되는 단어 (4회 이상)'
                })
            
            # 특정 패턴 검사 (더 엄격한 기준)
            patterns = [
                (r'또한|그리고|그러나|하지만|따라서|그리고|또한|그러나|하지만|따라서', '접속사 패턴'),
                (r'~하다|~이다|~되다|~하였다|~하였다고|~하였다는', '동사 패턴'),
                (r'~적|~적인|~적으로|~적이다|~적이었다', '형용사 패턴'),
                (r'~라고|~라고 한다|~라고 할 수 있다|~라고 생각한다', '인용 패턴'),
                (r'~것|~것이다|~것이라고|~것이라고 할 수 있다', '것 패턴')
            ]
            
            pattern_count = 0
            for pattern, desc in patterns:
                matches = list(re.finditer(pattern, sentence))
                if matches:
                    pattern_count += len(matches)
                    for match in matches:
                        reasons.append({
                            'type': 'pattern',
                            'text': match.group(),
                            'description': desc
                        })
            
            # 패턴 점수 계산 (더 엄격한 기준)
            if pattern_count > 0:
                score += min(0.4, pattern_count * 0.1)  # 최대 0.4까지
            
            return {
                'score': min(score, 1.0),
                'reasons': reasons
            }
        except Exception as e:
            logger.error(f"문장 패턴 분석 중 오류: {str(e)}")
            return {
                'score': 0.0,
                'reasons': []
            }
    
    def _split_into_sentences(self, text: str) -> list:
        """텍스트를 문장 단위로 분리합니다."""
        try:
            return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        except Exception as e:
            logger.error(f"문장 분리 중 오류: {str(e)}")
            return []

def main():
    killer = GPTKiller()
    
    # 테스트
    sample_text = "This is a sample text to test the GPT Killer."
    result = killer.analyze_text(sample_text)
    
    print(f"GPT 생성 확률: {result['gpt_probability']:.2%}")
    print(f"표절률: {result['plagiarism_rate']:.2%}")

if __name__ == "__main__":
    main() 