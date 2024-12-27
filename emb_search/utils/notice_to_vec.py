from konlpy.tag import Okt
from fasttext import load_model
import numpy as np
import re
from pydantic import BaseModel, Field, validator

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
