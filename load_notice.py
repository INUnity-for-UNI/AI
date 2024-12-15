from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine
from sqlalchemy.sql import text


# SSH 터널 설정
server = SSHTunnelForwarder(
    ('ssh.squidjiny.com', 22),  # SSH 서버 주소와 포트
    ssh_username='inunity',  # SSH 사용자 이름
    ssh_password='inunity1004',  # SSH 비밀번호 (SSH 키를 사용할 경우 제거)
    remote_bind_address=('127.0.0.1', 3306),  # 데이터베이스 호스트와 포트
    local_bind_address=('127.0.0.1', 3307)  # 로컬에서 접속할 포트
)

server.start()  # 터널 시작
print("시작");

# SQLAlchemy를 사용한 데이터베이스 연결
DATABASE_URL = "mysql+pymysql://root:1234@localhost:3307/inunity"
engine = create_engine(DATABASE_URL, echo = True)

# 데이터 조회
with engine.connect() as connection:
    result = connection.execute(text("SELECT * FROM notices")) 
    for row in result:
        print(row[3])

server.stop()  # 터널 종료