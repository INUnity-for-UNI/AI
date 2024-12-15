from konlpy.tag import Okt
from fasttext import load_model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
from functools import lru_cache


notices = []
with open('./DB/notices.txt', 'r') as file:
    for line in file:
        notices.append(line.strip())

# 1. 형태소 분석기 객체 생성 (공지사항 제목 -> 명사, 동사 등으로 변경)
okt = Okt()

# 2. FastText 모델 로드
fasttext_model_path = "./models/cc.ko.300.bin"  # 영어와 한국어를 포함하는 FastText 모델 경로
fasttext_model = load_model(fasttext_model_path)

# 3. 벡터 변환 함수
def words2vec(text):
    eng_text = re.findall(r'[a-zA-Z]+', text)
    eng_text = [e.lower() for e in eng_text]
    words = okt.nouns(text)
    words.extend(eng_text)
    vectors = []
    for word in words:
        vectors.append(fasttext_model.get_word_vector(word))
    return np.array(vectors)

def sentence2vec(sentence):
    vectors = words2vec(sentence)
    if len(vectors) == 0:  # 벡터가 비어있는 경우
        return np.zeros(300)
    return np.mean(vectors, axis=0)  # 300차원 vector return

# 4. 공지사항 데이터 벡터화
notice_vectors = np.load('./DB/notice_vectors.npy')

# 단어 벡터 얻기
query = '스타인유'

# 검색 시간 측정

query_vector = sentence2vec(query).reshape(1, -1)   # 이 단계에서 많은 시간 소요

similarities = cosine_similarity(query_vector, notice_vectors).flatten()

sorted_indices = np.argsort(-similarities)

top_k = 5  # 상위 검색 결과 설정
print("검색어:", query)
print("가장 유사한 공지사항:")
for rank, idx in enumerate(sorted_indices[:top_k], start=1):
    print(f"{rank}. {notices[idx]} (유사도: {similarities[idx]:.4f})")
