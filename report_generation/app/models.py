from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, DateTime, Text, String, DECIMAL

Base = declarative_base()

class ChatLog(Base):
    __tablename__ = "chat_logs"         # aligns with your DB
    chat_id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    question = Column(Text)
    answer = Column(Text)
    confidence_score = Column(DECIMAL(4,3))
    status = Column(String)
    created_at = Column(DateTime)

# We'll use 'admin_queue' (snake_case). If your DB currently uses adminQueue (camelCase),
# either create a view or adjust the table name here accordingly.
class AdminQueue(Base):
    __tablename__ = "admin_queue"
    queue_id = Column(Integer, primary_key=True)
    chat_id = Column(Integer)
    admin_id = Column(Integer, nullable=True)
    # assigned_at, resolved_at omitted for now



