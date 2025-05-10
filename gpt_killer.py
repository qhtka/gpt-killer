import re
from collections import Counter
import logging
import requests
from bs4 import BeautifulSoup
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
import nltk
import torch
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class GPTKiller:
    def __init__(self):
        self.search_cache = {}  # 검색 결과 캐시
        self.bert_model = None
        self.tokenizer = None
        self.sentence_transformer = None
        self.gpt2_model = None
        self.gpt2_tokenizer = None
        
        # 패턴 목록 정의
        self.logical_connectors = ['또한', '그리고', '그러나', '하지만', '따라서', '그러므로', '그래서', '결론적으로']
        self.passive_voice = ['~되다', '~받다', '~당하다', '~에 의해', '~에 의하여', '~로 인해', '~로 말미암아']
        self.adjective_patterns = ['~적', '~적인', '~적으로', '~스러운', '~스럽게']
        self.citation_patterns = ['~라고', '~라고 한다', '~라고 하였다', '~라고 말했다', '~라고 언급했다']
        self.vague_expressions = ['~것', '~것이다', '~것을', '~것이', '~것은', '~것으로', '~것에']
        
        # NLTK 데이터 다운로드
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            
        # 모델 초기화
        try:
            logger.info("모델 초기화 시작")
            
            # Sentence Transformer 모델 로드
            try:
                self.sentence_transformer = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("Sentence Transformer 모델 로드 완료")
            except Exception as e:
                logger.error(f"Sentence Transformer 모델 로드 실패: {str(e)}")
            
            # GPT-2 모델 로드
            try:
                self.gpt2_model = GPT2LMHeadModel.from_pretrained('gogamza/kobart-base-v2')
                self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gogamza/kobart-base-v2')
                logger.info("GPT-2 모델 로드 완료")
            except Exception as e:
                logger.error(f"GPT-2 모델 로드 실패: {str(e)}")
            
        except Exception as e:
            logger.error(f"모델 초기화 중 오류 발생: {str(e)}", exc_info=True)
            # 모델 초기화 실패 시에도 기본 기능은 동작하도록 함
            pass
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리를 수행합니다."""
        try:
            # 소문자 변환
            text = text.lower()
            
            # 특수문자 제거
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # 불용어 제거
            stop_words = set(stopwords.words('korean'))
            words = word_tokenize(text)
            words = [w for w in words if w not in stop_words]
            
            # 어간 추출
            stemmer = PorterStemmer()
            words = [stemmer.stem(w) for w in words]
            
            return ' '.join(words)
        except Exception as e:
            logger.error(f"텍스트 전처리 중 오류: {str(e)}")
            return text
    
    def _create_ngrams(self, text: str, n: int = 3) -> list:
        """n-gram을 생성합니다."""
        try:
            words = text.split()
            return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        except Exception as e:
            logger.error(f"n-gram 생성 중 오류: {str(e)}")
            return []
    
    def _calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """코사인 유사도를 계산합니다."""
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception as e:
            logger.error(f"코사인 유사도 계산 중 오류: {str(e)}")
            return 0.0
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """자카드 유사도를 계산합니다."""
        try:
            set1 = set(text1.split())
            set2 = set(text2.split())
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            logger.error(f"자카드 유사도 계산 중 오류: {str(e)}")
            return 0.0
    
    def _calculate_levenshtein_distance(self, text1: str, text2: str) -> float:
        """레벤슈타인 거리를 계산합니다."""
        try:
            if len(text1) < len(text2):
                return self._calculate_levenshtein_distance(text2, text1)
            if len(text2) == 0:
                return len(text1)
            
            previous_row = range(len(text2) + 1)
            for i, c1 in enumerate(text1):
                current_row = [i + 1]
                for j, c2 in enumerate(text2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            max_len = max(len(text1), len(text2))
            return 1 - (previous_row[-1] / max_len) if max_len > 0 else 0.0
        except Exception as e:
            logger.error(f"레벤슈타인 거리 계산 중 오류: {str(e)}")
            return 0.0
    
    def _calculate_bert_similarity(self, text1: str, text2: str) -> float:
        """BERT 기반 유사도를 계산합니다."""
        try:
            if not self.sentence_transformer:
                return 0.0
                
            embeddings1 = self.sentence_transformer.encode([text1])[0]
            embeddings2 = self.sentence_transformer.encode([text2])[0]
            
            similarity = np.dot(embeddings1, embeddings2) / (
                np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2)
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"BERT 유사도 계산 중 오류: {str(e)}")
            return 0.0
    
    def _check_plagiarism(self, text: str) -> dict:
        """텍스트의 표절 여부를 검사합니다."""
        try:
            sentences = self._split_into_sentences(text)
            plagiarism_details = []
            total_sentences = len(sentences)
            plagiarized_sentences = 0
            
            for sentence in sentences:
                # 5어절 이상인 문장만 검사 (기준 더 강화)
                eojeols = self._split_into_eojeols(sentence)
                if len(eojeols) >= 5:
                    # 문장 검색
                    search_results = self._search_sentence(sentence)
                    
                    if search_results:
                        # 각 검색 결과에 대해 다양한 유사도 계산
                        similarities = []
                        for result in search_results:
                            # 전처리
                            processed_sentence = self._preprocess_text(sentence)
                            processed_result = self._preprocess_text(result)
                            
                            # 다양한 유사도 계산
                            cosine_sim = self._calculate_cosine_similarity(processed_sentence, processed_result)
                            jaccard_sim = self._calculate_jaccard_similarity(processed_sentence, processed_result)
                            levenshtein_sim = self._calculate_levenshtein_distance(processed_sentence, processed_result)
                            bert_sim = self._calculate_bert_similarity(sentence, result)
                            
                            # n-gram 매칭
                            ngrams_sentence = self._create_ngrams(processed_sentence)
                            ngrams_result = self._create_ngrams(processed_result)
                            ngram_matches = len(set(ngrams_sentence).intersection(set(ngrams_result)))
                            ngram_sim = ngram_matches / max(len(ngrams_sentence), len(ngrams_result)) if ngrams_sentence and ngrams_result else 0
                            
                            # 종합 유사도 계산 (가중치 조정)
                            similarity = (
                                cosine_sim * 0.3 +
                                jaccard_sim * 0.2 +
                                levenshtein_sim * 0.2 +
                                bert_sim * 0.2 +
                                ngram_sim * 0.1
                            )
                            
                            similarities.append({
                                'text': result,
                                'similarity': similarity,
                                'details': {
                                    'cosine_similarity': cosine_sim,
                                    'jaccard_similarity': jaccard_sim,
                                    'levenshtein_similarity': levenshtein_sim,
                                    'bert_similarity': bert_sim,
                                    'ngram_similarity': ngram_sim
                                }
                            })
                        
                        # 유사도가 0.65 이상인 결과만 표절로 판정 (기준 더 강화)
                        high_similarity_results = [r for r in similarities if r['similarity'] >= 0.65]
                        if high_similarity_results:
                            plagiarized_sentences += 1
                            plagiarism_details.append({
                                'sentence': sentence,
                                'matches': high_similarity_results
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
    
    def _analyze_sentence_pattern(self, sentence: str) -> dict:
        """문장의 패턴을 분석합니다."""
        try:
            reasons = []
            score = 0.0
            
            # 문장 길이 분석 (60단어 이상으로 기준 완화)
            words = word_tokenize(sentence)
            if len(words) > 60:
                reasons.append({
                    'type': 'long_sentence',
                    'text': sentence,
                    'description': '문장이 너무 깁니다.',
                    'start': 0,
                    'end': len(sentence)
                })
                score += 0.05
            
            # 반복 단어 분석 (5회 이상으로 기준 완화)
            word_freq = Counter(words)
            repeated_words = [word for word, count in word_freq.items() if count > 5]
            if repeated_words:
                for word in repeated_words:
                    # 반복된 단어의 모든 위치 찾기
                    start = 0
                    while True:
                        start = sentence.find(word, start)
                        if start == -1:
                            break
                        reasons.append({
                            'type': 'repeated_words',
                            'text': word,
                            'description': f'반복되는 단어: {word}',
                            'start': start,
                            'end': start + len(word)
                        })
                        start += len(word)
                score += 0.05
            
            # 패턴 분석 (4회 이상으로 기준 완화)
            patterns = [
                (r'또한|그리고|그러나|하지만|따라서', 'logical_connector', '논리적 연결어'),
                (r'[가-힣]+하다|[가-힣]+이다|[가-힣]+되다', 'passive_voice', '수동태'),
                (r'[가-힣]+적|[가-힣]+적인|[가-힣]+적으로', 'adjective_pattern', '형용사 패턴'),
                (r'[가-힣]+라고|[가-힣]+라고 한다', 'citation_pattern', '인용 패턴'),
                (r'[가-힣]+것|[가-힣]+것이다', 'vague_expression', '모호한 표현')
            ]
            
            for pattern, pattern_type, description in patterns:
                matches = list(re.finditer(pattern, sentence))
                if len(matches) > 4:  # 4번 이상 반복될 때만 감지
                    for match in matches:
                        reasons.append({
                            'type': pattern_type,
                            'text': match.group(),
                            'description': f'{description}: {match.group()}',
                            'start': match.start(),
                            'end': match.end()
                        })
                    score += 0.03
            
            # 시작 위치 기준으로 정렬
            reasons.sort(key=lambda x: x['start'])
            
            return {
                'score': min(0.9, max(0.1, score)),
                'reasons': reasons
            }
        except Exception as e:
            logger.error(f"문장 패턴 분석 중 오류: {str(e)}")
            return {'score': 0.3, 'reasons': []}

    def analyze_text(self, text):
        """텍스트 분석"""
        try:
            # 문장 분리
            sentences = self._split_into_sentences(text)
            print(f"분리된 문장 수: {len(sentences)}")
            
            # 각 문장 분석
            sentence_analyses = []
            total_score = 0
            
            for sentence in sentences:
                print(f"\n분석 중인 문장: {sentence}")
                reasons = []
                sentence_score = 0
                
                # 1. 긴 문장 체크 (150자 이상으로 완화)
                if len(sentence) > 150:
                    reasons.append({
                        'type': 'long_sentence',
                        'text': sentence,
                        'description': '문장이 너무 깁니다.',
                        'start': 0,
                        'end': len(sentence)
                    })
                    sentence_score += 5  # 점수 감소
                
                # 2. 반복된 단어 체크 (5회 이상으로 완화)
                words = sentence.split()
                word_count = {}
                for word in words:
                    if len(word) > 1:  # 한 글자 단어 제외
                        word_count[word] = word_count.get(word, 0) + 1
                
                for word, count in word_count.items():
                    if count >= 5:  # 기준 완화
                        start = sentence.find(word)
                        reasons.append({
                            'type': 'repeated_words',
                            'text': word,
                            'description': f'"{word}"가 {count}번 반복되었습니다.',
                            'start': start,
                            'end': start + len(word)
                        })
                        sentence_score += 3  # 점수 감소
                
                # 3. 논리적 연결어 체크 (3회 이상으로 완화)
                connector_count = 0
                for connector in self.logical_connectors:
                    if connector in sentence:
                        connector_count += 1
                        start = sentence.find(connector)
                        reasons.append({
                            'type': 'logical_connector',
                            'text': connector,
                            'description': f'논리적 연결어 "{connector}"가 사용되었습니다.',
                            'start': start,
                            'end': start + len(connector)
                        })
                
                if connector_count >= 3:  # 기준 완화
                    sentence_score += 3  # 점수 감소
                
                # 4. 수동태 체크 (3회 이상으로 완화)
                passive_count = 0
                for passive in self.passive_voice:
                    if passive in sentence:
                        passive_count += 1
                        start = sentence.find(passive)
                        reasons.append({
                            'type': 'passive_voice',
                            'text': passive,
                            'description': f'수동태 "{passive}"가 사용되었습니다.',
                            'start': start,
                            'end': start + len(passive)
                        })
                
                if passive_count >= 3:  # 기준 완화
                    sentence_score += 3  # 점수 감소
                
                # 5. 형용사 패턴 체크 (3회 이상으로 완화)
                pattern_count = 0
                for pattern in self.adjective_patterns:
                    if pattern in sentence:
                        pattern_count += 1
                        start = sentence.find(pattern)
                        reasons.append({
                            'type': 'adjective_pattern',
                            'text': pattern,
                            'description': f'형용사 패턴 "{pattern}"이 사용되었습니다.',
                            'start': start,
                            'end': start + len(pattern)
                        })
                
                if pattern_count >= 3:  # 기준 완화
                    sentence_score += 3  # 점수 감소
                
                # 6. 인용 패턴 체크 (3회 이상으로 완화)
                citation_count = 0
                for pattern in self.citation_patterns:
                    if pattern in sentence:
                        citation_count += 1
                        start = sentence.find(pattern)
                        reasons.append({
                            'type': 'citation_pattern',
                            'text': pattern,
                            'description': f'인용 패턴 "{pattern}"이 사용되었습니다.',
                            'start': start,
                            'end': start + len(pattern)
                        })
                
                if citation_count >= 3:  # 기준 완화
                    sentence_score += 3  # 점수 감소
                
                # 7. 모호한 표현 체크 (3회 이상으로 완화)
                vague_count = 0
                for expression in self.vague_expressions:
                    if expression in sentence:
                        vague_count += 1
                        start = sentence.find(expression)
                        reasons.append({
                            'type': 'vague_expression',
                            'text': expression,
                            'description': f'모호한 표현 "{expression}"이 사용되었습니다.',
                            'start': start,
                            'end': start + len(expression)
                        })
                
                if vague_count >= 3:  # 기준 완화
                    sentence_score += 3  # 점수 감소
                
                if reasons:
                    sentence_analyses.append({
                        'sentence': sentence,
                        'reasons': reasons,
                        'score': sentence_score
                    })
                    total_score += sentence_score
            
            print(f"분석된 문장 수: {len(sentence_analyses)}")
            print(f"총 점수: {total_score}")
            
            # GPT 점수 계산 (0-100, 기준 완화)
            gpt_score = min(100, total_score * 0.7)  # 30% 감소
            
            # 인간 점수 계산 (0-100)
            human_score = max(0, 100 - gpt_score)
            
            return {
                'gpt_score': gpt_score,
                'human_score': human_score,
                'sentence_analyses': sentence_analyses
            }
            
        except Exception as e:
            print(f"분석 중 오류 발생: {str(e)}")
            return {
                'gpt_score': 0,
                'human_score': 100,
                'sentence_analyses': []
            }
    
    def _calculate_perplexity(self, text: str) -> float:
        """텍스트의 perplexity를 계산합니다."""
        try:
            if not self.gpt2_model or not self.gpt2_tokenizer:
                return 0.3  # 기본값
                
            # 문장 단위로 분리
            sentences = sent_tokenize(text)
            total_perplexity = 0.0
            valid_sentences = 0
            
            for sentence in sentences:
                # 토큰화
                inputs = self.gpt2_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
                
                # perplexity 계산
                with torch.no_grad():
                    outputs = self.gpt2_model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()
                    
                    if perplexity < 1000:  # 비정상적으로 높은 값 제외
                        total_perplexity += perplexity
                        valid_sentences += 1
            
            avg_perplexity = total_perplexity / valid_sentences if valid_sentences > 0 else 0.0
            # 0.3~0.7 범위로 정규화
            normalized_score = 0.3 + (min(1.0, avg_perplexity / 200) * 0.4)  # 기준값 200으로 완화
            return normalized_score
        except Exception as e:
            logger.error(f"Perplexity 계산 중 오류: {str(e)}")
            return 0.3  # 기본값
    
    def _calculate_burstiness(self, text: str) -> float:
        """텍스트의 burstiness를 계산합니다."""
        try:
            # 문장 단위로 분리
            sentences = sent_tokenize(text)
            if not sentences:
                return 0.3  # 기본값
            
            # 문장 길이의 표준편차
            lengths = [len(s.split()) for s in sentences]
            length_std = np.std(lengths)
            length_mean = np.mean(lengths)
            
            # 문장별 perplexity의 표준편차
            perplexities = [self._calculate_perplexity(s) for s in sentences]
            perplexity_std = np.std(perplexities)
            perplexity_mean = np.mean(perplexities)
            
            # 문장 구조의 다양성
            structures = []
            for sentence in sentences:
                # 품사 태깅
                pos_tags = pos_tag(word_tokenize(sentence))
                # 품사 패턴 추출
                structure = ' '.join([tag for word, tag in pos_tags])
                structures.append(structure)
            
            # 구조 패턴의 다양성
            unique_structures = len(set(structures))
            structure_diversity = unique_structures / len(structures)
            
            # 종합 burstiness 점수 계산
            burstiness = (
                (length_std / length_mean) * 0.4 +  # 문장 길이 다양성
                (perplexity_std / perplexity_mean) * 0.3 +  # perplexity 다양성
                structure_diversity * 0.3  # 구조 다양성
            )
            
            # 0.3~0.7 범위로 정규화
            normalized_score = 0.3 + (min(1.0, burstiness) * 0.4)
            return normalized_score
        except Exception as e:
            logger.error(f"Burstiness 계산 중 오류: {str(e)}")
            return 0.3  # 기본값
    
    def _analyze_ngram_patterns(self, text: str) -> float:
        """n-gram 패턴을 분석합니다."""
        try:
            # 문장 단위로 분리
            sentences = sent_tokenize(text)
            if not sentences:
                return 0.3  # 기본값
            
            # 2-gram, 3-gram, 4-gram 생성
            bigrams = []
            trigrams = []
            fourgrams = []
            
            for sentence in sentences:
                words = word_tokenize(sentence)
                bigrams.extend([' '.join(words[i:i+2]) for i in range(len(words)-1)])
                trigrams.extend([' '.join(words[i:i+3]) for i in range(len(words)-2)])
                fourgrams.extend([' '.join(words[i:i+4]) for i in range(len(words)-3)])
            
            # 각 n-gram의 반복률 계산
            def calculate_repetition_rate(ngrams):
                if not ngrams:
                    return 0.0
                counter = Counter(ngrams)
                total = len(ngrams)
                unique = len(counter)
                return 1 - (unique / total)
            
            bigram_repetition = calculate_repetition_rate(bigrams)
            trigram_repetition = calculate_repetition_rate(trigrams)
            fourgram_repetition = calculate_repetition_rate(fourgrams)
            
            # 일반적인 패턴 분석
            common_patterns = [
                r'또한|그리고|그러나|하지만|따라서',
                r'~하다|~이다|~되다',
                r'~적|~적인|~적으로',
                r'~라고|~라고 한다',
                r'~것|~것이다'
            ]
            
            pattern_matches = sum(
                len(re.findall(pattern, text))
                for pattern in common_patterns
            )
            
            # 종합 점수 계산
            ngram_score = (
                bigram_repetition * 0.2 +
                trigram_repetition * 0.3 +
                fourgram_repetition * 0.3 +
                (pattern_matches / len(sentences)) * 0.2
            )
            
            # 0.3~0.7 범위로 정규화
            normalized_score = 0.3 + (min(1.0, ngram_score) * 0.4)
            return normalized_score
        except Exception as e:
            logger.error(f"N-gram 패턴 분석 중 오류: {str(e)}")
            return 0.3  # 기본값
    
    def _analyze_mechanical_style(self, text: str) -> float:
        """기계적 문체를 분석합니다."""
        try:
            # 문장 단위로 분리
            sentences = sent_tokenize(text)
            if not sentences:
                return 0.3  # 기본값
            
            # 감정 표현 분석
            emotional_expressions = [
                r'~행복|~기쁨|~슬픔|~화가|~걱정',
                r'~좋다|~나쁘다|~힘들다|~쉽다',
                r'~싫다|~좋아하다|~싫어하다',
                r'~감사|~미안|~부끄럽'
            ]
            
            # 논리적 연결어 분석
            logical_connectors = [
                r'결론적으로|따라서|그러므로|그래서',
                r'또한|그리고|그러나|하지만',
                r'중요한 것은|특히|특별히',
                r'~것은|~것이|~것을'
            ]
            
            # 수동태 분석
            passive_voice = [
                r'~되다|~받다|~당하다',
                r'~에 의해|~에 의하여',
                r'~로 인해|~로 말미암아'
            ]
            
            # 각 패턴의 빈도 계산
            emotional_count = sum(len(re.findall(pattern, text)) for pattern in emotional_expressions)
            logical_count = sum(len(re.findall(pattern, text)) for pattern in logical_connectors)
            passive_count = sum(len(re.findall(pattern, text)) for pattern in passive_voice)
            
            # 문법 오류 분석
            grammar_errors = 0
            for sentence in sentences:
                # 간단한 문법 오류 검사 (예: 조사 중복, 어색한 어순 등)
                if re.search(r'은은|는는|이이|가가|을을|를를', sentence):
                    grammar_errors += 1
                if re.search(r'~하다하다|~되다되다', sentence):
                    grammar_errors += 1
            
            # 종합 점수 계산
            mechanical_score = (
                (1 - min(1, emotional_count / len(sentences))) * 0.3 +  # 감정 표현 부족
                (min(1, logical_count / len(sentences))) * 0.3 +  # 논리적 연결어 과다
                (min(1, passive_count / len(sentences))) * 0.2 +  # 수동태 과다
                (1 - min(1, grammar_errors / len(sentences))) * 0.2  # 문법 오류 부족
            )
            
            # 0.3~0.7 범위로 정규화
            normalized_score = 0.3 + (min(1.0, mechanical_score) * 0.4)
            return normalized_score
        except Exception as e:
            logger.error(f"기계적 문체 분석 중 오류: {str(e)}")
            return 0.3  # 기본값
    
    def _calculate_final_score(self, scores: dict) -> float:
        """최종 GPT 생성 확률을 계산합니다."""
        try:
            weights = {
                'perplexity': 0.25,     # Perplexity (25%)
                'burstiness': 0.25,     # Burstiness (25%)
                'ngram_patterns': 0.25, # N-gram 패턴 (25%)
                'mechanical_style': 0.25 # 기계적 문체 (25%)
            }
            
            # 각 점수를 0.1~0.9 범위로 조정하여 더 관대하게 만듦
            adjusted_scores = {}
            for category, score in scores.items():
                # 0.1~0.9 범위로 조정
                adjusted_score = 0.1 + (score * 0.8)
                adjusted_scores[category] = adjusted_score
            
            final_score = sum(
                adjusted_scores[category] * weight
                for category, weight in weights.items()
            )
            
            return min(0.8, max(0.2, final_score))  # 20%~80% 범위로 제한
        except Exception as e:
            logger.error(f"최종 점수 계산 중 오류: {str(e)}")
            return 0.3  # 기본값
    
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
    
    print(f"GPT 생성 확률: {result['gpt_score']:.2%}")
    print(f"인간 생성 확률: {result['human_score']:.2%}")

if __name__ == "__main__":
    main() 