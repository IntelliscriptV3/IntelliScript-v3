import os
from fastapi import FastAPI
from pydantic import BaseModel
from app.agent import chat

app = FastAPI(title="IntelliScript Reports")

class ChatRequest(BaseModel):
    user_id: int
    message: str

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    return chat(req.user_id, req.message)
