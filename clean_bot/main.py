from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import torch

def load_text_filter(model_name='smilegate-ai/kor_unsmile', device=0):
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores = True,
        function_to_apply='sigmoid'
    )
    return pipe

def filter_text(text, pipe, threshold=0.3):
    results = pipe(text)[0]

    # clean 점수 찾기
    clean_score = next(item['score'] for item in results if item['label'] == 'clean')

    # 이진 분류 결과 (clean: True, non-clean: False)
    if clean_score <= threshold:
        filtering = True
    else:
        filtering = False

    # 결과 반환
    return {
        'binary_class': filtering,
        'clean_score': clean_score,
        'filtered_text': text if clean_score >= threshold else "부적절한 내용으로 필터링 되었습니다.",
        'original_text': text,
        'total': results
    }

# 사용 예시
if __name__ == "__main__":
    # 필터 로드
    text_filter = load_text_filter()

    # 테스트할 텍스트 목록
    test_texts = [
        "전라도 사람"
    ]

    # 각 텍스트에 대해 필터링 수행
    for text in test_texts:
        if "전라도" in text:
            text = text.replace("전라도", "")
            print(text)
        result = filter_text(text, text_filter)

        print("\n=== 필터링 결과 ===")
        print(f"원본 텍스트: {result['original_text']}")
        if result['binary_class']:
            print(f"이진 분류 결과: 부적절, {result['clean_score']}")
        else:
            print(f"이진 분류 결과: Clean, {result['clean_score']}")
        print(f"필터링 결과 --> {result['filtered_text']}")