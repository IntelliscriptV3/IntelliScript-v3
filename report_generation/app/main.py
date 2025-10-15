import os
from fastapi import FastAPI
from pydantic import BaseModel
from .agent import chat

# app = FastAPI(title="IntelliScript Reports")


# @app.post("/chat")
class ReportGeneration:
    def __init__(self, user_id):
        self.user_id = user_id

    def chat_endpoint(self, message: str):
        return chat(self.user_id, message)

if __name__ == "__main__":
    report_gen = ReportGeneration(user_id=1)
    response = report_gen.chat_endpoint("Show me the attendance trends for the last semester.")
    print(response)