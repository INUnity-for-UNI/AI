from konlpy.tag import Okt
from fasttext import load_model
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field, validator
from tqdm import tqdm

# VectorizeService 클래스
class VectorizeService:
    def __init__(self, model_path: str, okt=None):
        self.fasttext_model = load_model(model_path)
        self.okt = okt if okt else Okt()

    def clean_text(self, text: str):
        text = re.sub(r'[^\w\s]', '', text)  # 구두점 제거
        return text.lower()  # 소문자로 변환

    def words2vec(self, text: str):
        text = self.clean_text(text)
        eng_text = re.findall(r'[a-zA-Z]+', text)
        eng_text = [e.lower() for e in eng_text]
        words = self.okt.nouns(text)
        words.extend(eng_text)
        vectors = [self.fasttext_model.get_word_vector(word) for word in words]
        return np.array(vectors)

    def sentence2vec(self, sentence: str):
        vectors = self.words2vec(sentence)
        return vectors

# Query 클래스
class Query(BaseModel):
    text: str = Field(..., description="사용자가 입력한 검색어")

    @validator("text")
    def validate_text(cls, value):
        if not value or not isinstance(value, str):
            raise ValueError("검색어는 비어 있을 수 없으며 문자열이어야 합니다.")
        return value

    def to_vector(self, vectorizer: VectorizeService):
        return vectorizer.sentence2vec(self.text)

# 메인 코드
if __name__ == '__main__':
    # 공지사항 불러오기
    notices = []
    with open('./DB/notices.txt', 'r') as file:
        for line in file:
            notices.append(line.strip())

    # VectorizeService 초기화
    model_path = './models/cc.ko.300.bin'
    vectorizer = VectorizeService(model_path=model_path)

    # 공지사항 제목 벡터화
    print('\nStart Vectorization')
    notice_vectors = [vectorizer.sentence2vec(notice) for notice in tqdm(notices)]
    
    # 사용자 검색어 처리
    query = Query(text='STARinU')
    query_vectors = query.to_vector(vectorizer)
    print(query_vectors.shape)
    
    # 유사도 계산
    similarities = []
    for notice_vecs in notice_vectors:
        if len(notice_vecs) == 0:  # 공지사항에 명사가 없을 경우
            similarities.append(0)
            continue

        # 각 단어 벡터에 대한 유사도 계산
        word_similarities = []
        for query_vector in query_vectors:
            word_sim = cosine_similarity(query_vector.reshape(1, -1), notice_vecs).max()
            word_similarities.append(word_sim)

        # 최대 유사도의 평균 계산
        avg_similarity = np.mean(word_similarities)
        similarities.append(avg_similarity)
    
    # 가장 유사한 공지사항 찾기 (상위 10개)
    top_k_indices = np.argsort(similarities)[-10:][::-1]
    top_k_notices = [notices[i] for i in top_k_indices]

    print("검색어:", query.text)
    print("가장 유사한 공지사항 제목 10개:")
    for i, index in enumerate(top_k_indices, 1):
        print(f"{i}. {notices[index]} (유사도 점수: {similarities[index]:.4f})")
