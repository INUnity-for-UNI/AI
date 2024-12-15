import re
import time
import pickle
from gensim import models
from konlpy.tag import Hannanum, Okt, Kkma, Mecab
from sklearn.metrics.pairwise import cosine_similarity

hannanum = Okt()


# 모델 불러오는 함수
def load_model(file_path):
    print("model loading ...")
    start = time.time()
    ko_model = models.fasttext.load_facebook_model(file_path)
    print(f"Loading time {time.time() - start}")

    return ko_model



# 학교 공지사항 제목 로드 함수
def load_notice():
    notice = [
        '정보기술대 2024년도 METLAB 수상팀 공지',
        '커스텀 SNS 경진대회 예선심사 결과발표',
        '정보기술대학 프로그램 및 CJ교육과정 설명회 자료',
        '정보기술대학 코드페스티벌 안내',
        '[정보기술대학] 정보대 콘테SW트 수기공모 안내',
    ]
    return notice


# 문장의 명사 추출 함수
def get_nouns(text):
    eng_text = re.findall(r'[a-zA-Z]+', text)
    nouns = hannanum.nouns(text)
    nouns.extend(eng_text)  # 'eng_res'를 'eng_text'로 수정
    return nouns
    

# 문장의 명사들을 벡터 값으로 변환하는 함수
def nouns_to_vector(nouns):
    vectors = []
    for noun in nouns:
        try:
            vectors.append(ko_model.wv[noun])
        except KeyError:
            pass

    return vectors


# 벡터 저장 함수 (공지사항 제목을 미리 벡터화 후 DB에 저장)
def save_vectors(vectors, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vectors, f)


# 벡터 로드 함수 (미리 벡터화된 공지사항을 DB에서 load)
def load_vectors(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    ko_model = load_model('./models/cc.ko.300.bin')
    notices = load_notice()
    all_vectors = []

    for ntc in notices:
        nouns = get_nouns(ntc)
        print(nouns)
        vectors = nouns_to_vector(nouns)
        all_vectors.append(vectors)
    
    save_vectors(all_vectors, './DB/notice_vectors.pkl')
    
    # 저장된 벡터 확인
    loaded_vectors = load_vectors('./DB/notice_vectors.pkl')

    with open('./DB/notice_vectors.txt', 'w') as f:
        f.write(str(loaded_vectors))

    print(f"저장된 공지사항 수: {len(loaded_vectors)}")
    for i, vectors in enumerate(loaded_vectors):
        print(f"공지사항 {i+1}: {len(vectors)}개의 벡터")


def cosine_sim(query_vec, doc_vec):
    return 1 + cosine_similarity(query_vec.reshape(1, -1), doc_vec.reshape(1, -1))[0][0]