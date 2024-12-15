from dotenv import load_dotenv
import os
import time
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv('API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

content = open('DB/notice3.txt', 'r').read()

start_time = time.time()

user_prompt = f"""
    모든 대답은 한국어(문어체)로 대답해줘.
    아래 게시글 내용을 요약해줘.
    만약 게시글에 날짜와 장소에 대한 정보가 포함되어 있다면, 다음 형식으로 요약해줘:
    
    날짜: xxxx년 xx월 xx일
    장소: xx시 xx동
    내용: "핵심 내용 간략 설명"
    
    날짜와 장소 정보가 없으면, 간단히 핵심 내용만 요약해줘.
    게시글: [{content}]
"""

response = model.generate_content(
    user_prompt,
    generation_config=genai.types.GenerationConfig(
    # Only one candidate for now.
    candidate_count=1,
    stop_sequences=['x'],
    #max_output_tokens=40,
    temperature=1.0)
)
print(response.text)
end_time = time.time()
execution_time = end_time - start_time
print(f"실행 시간: {execution_time:.2f} 초")