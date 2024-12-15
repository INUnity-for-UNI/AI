from konlpy.tag import Okt
from fasttext import load_model
import numpy as np
from tqdm import tqdm
import re

# 1. INU 공지사항 제목 불러오기
notices = []
with open('./DB/notices.txt', 'r') as file:
    for line in file:
        notices.append(line.strip())

# 2. 형태소 분석기 객체 생성 (공지사항 제목 -> 명사, 어절 등으로 변경)
okt = Okt()

# 3. FastText 모델 로드
fasttext_model_path = "./models/cc.ko.300.bin"  # FastText 이진 모델 경로
fasttext_model = load_model(fasttext_model_path)

# 4. 벡터 변환 함수
def words2vec(text):
    eng_text = re.findall(r'[a-zA-Z]+', text)       # 영어 검색 처리
    eng_text = [e.lower() for e in eng_text]        # 소문자 처리
    words = okt.phrases(text)                       # 어절 단위 전처리
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

# 5. 공지사항 데이터 벡터화
print('\nStart Vectorization')
notice_vectors = [sentence2vec(notice) for notice in tqdm(notices)]

np.save('./DB/notice_vectors.npy', notice_vectors)