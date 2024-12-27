import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from utils.emb_search import VectorizeService, Query

# 메인 코드
if __name__ == '__main__':
    # 공지사항 벡터 불러오기
    notice_vectors = np.load('./DB/notice_vectors.npy', allow_pickle=True)

    # 공지사항 불러오기
    notices = []
    with open('./DB/notices.txt', 'r') as file:
        for line in file:
            notices.append(line.strip())

    # VectorizeService 초기화
    model_path = './models/cc.ko.300.bin'
    vectorizer = VectorizeService(model_path=model_path)

    # 사용자 검색어 처리
    query = Query(text='STARinU')
    query_vectors = query.to_vector(vectorizer)

    # 유사도 계산
    similarities = []
    for notice_vecs in notice_vectors:
        if len(notice_vecs) == 0:  # 공지사항에 명사가 없을 경우 (이런 일 없겠지만 발생하면 귀찮으니)
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
