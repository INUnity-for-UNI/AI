from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
from pydantic import BaseModel, Field
import torch.nn.functional as F
import torch


# pydantic 정의
class FilterResult(BaseModel):
    binary_class: bool = Field(..., description="텍스트가 clean인지 여부")
    clean_score: float = Field(..., description="clean 점수")
    filtered_text: str = Field(..., description="필터링된 텍스트")
    original_text: str = Field(..., description="원본 텍스트")
    # total: list = Field(..., description="모든 라벨의 점수 리스트")


# 모델 호출 함수
def load_text_filter(model_name='smilegate-ai/kor_unsmile', device=0):
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=True,
        function_to_apply='sigmoid'
    )
    return pipe


# 필터링 함수
def filter_text(text, pipe, threshold=0.3) -> FilterResult:
    results = pipe(text)[0]

    # clean 점수 찾기
    clean_score = next(item['score'] for item in results if item['label'] == 'clean')

    # 이진 분류 결과 (clean: True, non-clean: False)
    binary_class = clean_score <= threshold

    # Pydantic 모델로 결과 반환
    return FilterResult(
        binary_class=binary_class,
        clean_score=clean_score,
        filtered_text=text if clean_score >= threshold else "부적절한 내용으로 필터링 되었습니다.",
        original_text=text,
        # total=results
    )


if __name__ == "__main__":
    # 필터 로드
    text_filter = load_text_filter()

    # 테스트할 텍스트 목록
    test_texts = [
        "전라도 사람"
    ]

    # 각 텍스트에 대해 필터링 수행
    for text in test_texts:
        # 특정 단어 제거 예시
        if "전라도" in text:
            text = text.replace("전라도", "")

        # 필터링 수행
        result = filter_text(text, text_filter)

        # 결과 출력
        print("\n=== 필터링 결과 ===")
        print(f"원본 텍스트: {result.original_text}")
        if result.binary_class:
            print(f"이진 분류 결과: 부적절, {result.clean_score}")
        else:
            print(f"이진 분류 결과: Clean, {result.clean_score}")
        print(f"필터링 결과 --> {result.filtered_text}")