from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os
import time
import google.generativeai as genai

# 환경 변수 로드
load_dotenv()

# Google Generative AI 초기화
genai.configure(api_key=os.getenv("API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")


# Pydantic 모델 정의
class NoticeSummaryRequest(BaseModel):
    content: str = Field(..., description="게시글 원문")
    language: str = Field("한국어(문어체)", description="요약에 사용할 언어")
    format_with_date_and_location: str = Field(
        """
        날짜: xxxx년 xx월 xx일
        장소: xx시 xx동
        내용: "핵심 내용 간략 설명"
        """,
        description="날짜와 장소 정보가 포함된 요약 형식",
    )


class NoticeSummaryResponse(BaseModel):
    summary: str = Field(..., description="요약된 게시글 내용")
    # execution_time: float = Field(..., description="요약 수행 시간 (초)")


# 텍스트 요약 요청 생성 함수
def create_summary_request(content: str) -> NoticeSummaryRequest:
    return NoticeSummaryRequest(content=content)


# 요약 수행 함수
def summarize_notice(request: NoticeSummaryRequest) -> NoticeSummaryResponse:
    user_prompt = f"""
    모든 대답은 {request.language}로 대답해줘.
    아래 게시글 내용을 요약해줘.

    1. 만약 게시글에 날짜와 장소에 대한 정보가 둘 다 포함되어 있다면:
        날짜: xxxx년 xx월 xx일
        장소: xx시 xx동
        내용: "핵심 내용 간략 설명" 

    2. 만약 게시글에 날짜 정보만 포함되어 있고, 장소 정보가 없다면:
        날짜: xxxx년 xx월 xx일
        내용: "핵심 내용 간략 설명"

    3. 만약 게시글에 날짜와 장소 정보가 둘 다 없다면:
        내용: "핵심 내용 간략 설명"

    게시글: [{request.content}]
    """

    start_time = time.time()

    # Generative AI 호출
    response = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            stop_sequences=["x"],
            temperature=1.0,
        ),
    )

    end_time = time.time()
    execution_time = end_time - start_time

    return NoticeSummaryResponse(
        summary=response.text.strip(),
        # execution_time=execution_time
        )


# 메인 함수
if __name__ == "__main__":
    # 요약 요청 생성
    content = '''
                ‼️⭐️법과대학 제36회 형사 모의재판 공지⭐️‼️
                안녕하세요, 학우 여러분 !
                법과대학 제27대 학생회 로운입니다🫶

                법과대학 최대행사인 💥형사 모의재판💥에 여러분을 초대합니다 !

                모의재판 이후 추첨이벤트가 진행되오니 많은 관심과 참여 부탁드립니다🔥🥰

                📌모의재판 일시 및 장소
                ▪️일시: 2024. 11. 21. (목) 16:00 ~
                ▪️장소: 인천대학교 23호관 대강당

                📌모의재판 세부일정
                15:00 ~ 16:00 입장
                16:00 ~ 16:10 개회식&학부장님 축사
                16:10 ~ 17:20 모의재판
                17:20 ~ 17:30 모의재판 강평
                17:30 ~ 17:40 출연진 인사
                17:40 ~ 18:00 추첨이벤트
                ※대강당 입장시에는 안전상의 문제가 발생할 수 있으므로 반드시 학생회의 통제에 따라주시기 바랍니다.

                📌모의재판 관련 세부사항
                ▪️참석하는 모든 인원에게(타과생 포함) 공결문이 제공됩니다.
                ※ 공결문 인정 시간: 15:00 ~
                ▪️추첨이벤트 번호표는 입장 시 배부되고, 모의재판 이후 추첨 행사가 진행됩니다.
                ▪️행사 당일 15:00부터 입장 가능합니다.
                ※ 행사장 분위기를 위해 잦은 출입은 자제하여 주시기 바랍니다.
                ▪️별도의 사전 신청 없이 당일 입장 가능합니다.

                📌 관련문의
                법과대학 학생회장 강민석 : 010-5787-2385
                법과대학 오픈채팅
    '''

    request = create_summary_request(content)

    # 요약 수행
    response = summarize_notice(request)

    # 결과 출력
    print("\n=== 요약 결과 ===")
    print(response.summary)
    # print(f"\n실행 시간: {response.execution_time:.2f} 초")