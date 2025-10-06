import os
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
import json
from vector_kb import FAISSKnowledgeBase
import os

class CRMSQLModule:
    def __init__(self, 
                 user_id: int,
                 db_url = None, 
                 api_key = None
                 ):

        self.user_id = user_id
        
        # Get database URL from environment variable or use default
        if db_url is None:
            db_url = os.getenv("DB_URL")
        
        # Get API key from environment variable
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # --- Setup DB session ---
        self.engine = create_engine(db_url)
        SessionLocal = sessionmaker(bind=self.engine)
        self.session = SessionLocal()

        self.faiss_kb = FAISSKnowledgeBase()
        self.faiss_kb.build_faiss_index()
        # if os.path.exists(self.faiss_kb.index_file) and os.path.exists(self.faiss_kb.metadata_file):
        #     self.faiss_kb.load_index()
        #     print("Loaded existing FAISS index.")
        # else:
        #     self.faiss_kb.build_faiss_index()

        os.environ["OPENAI_API_KEY"] = api_key
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        self.db = SQLDatabase.from_uri(db_url)

        agent_executor = create_sql_agent(self.llm, db=self.db, agent_type="tool-calling", verbose=True)
        self.agent_executor = agent_executor

    def __eval_agent(self, query: str, raw_answer: str):
        llm_eval_prompt = f"""
                The user asked: "{query}"
                The database agent responded: "{raw_answer}"

                Please do the following:
                1. Give a confidence score between 0.0 and 1.0 indicating
                how likely this answer is correct based on the database content.
                2. If database agent response is telling that it couldn't find an answer from the database, give it a lower confidence score near zero otherwise give it a higher score above 0.5.
                3. Provide a short reasoning for your score.

                Return only a JSON object like:
                {{"confidence": 0.85, "reason": "Explanation here"}}
                """

        eval_response = self.llm.invoke(llm_eval_prompt).content

        try:
            confidence_data = json.loads(eval_response)
            confidence_score = float(confidence_data.get("confidence", 0.0))
            explanation = confidence_data.get("reason", "")
            print(f"[eval_agent] confidence score: {confidence_score}, Explanation: {explanation}")
        except Exception:
            confidence_score = 0.0

        return confidence_score

    def __interface_agent(self, prompt: str):
        llm_response = self.llm.invoke(prompt)
        return llm_response.content

    def __chat_logs_model(self, user_id: int, query: str, final_answer: str, confidence_score: float, status: str):
        insert_query = text("""
                INSERT INTO chat_logs (user_id, question, answer, confidence_score, status, created_at)
                VALUES (:user_id, :question, :answer, :confidence_score, :status, :created_at)
                """)

        self.session.execute(insert_query, {
            "user_id": user_id,
            "question": query,
            "answer": final_answer,
            "confidence_score": confidence_score,
            "status": status,
            "created_at": datetime.now()
        })
        self.session.commit()

    def chat(self, query: str):
        """Function to handle SQL queries using the agent executor, 
        return user-friendly answers, and log results in chat_logs."""
        confidence_score = 0.0
        status = "failed"
        final_answer = ""

        try:
            # Step 1: Get raw response from SQL agent
            response = self.agent_executor.invoke({"input": query})
            raw_answer = response.get("output")

            print(f"[sql_rag] Raw answer: {raw_answer}")

            confidence_score = self.__eval_agent(query, raw_answer)
            print(f"[sql_rag] Confidence score: {confidence_score}")

            if confidence_score > 0.5:
                # Step 2: High confidence
                status = "answered"

                # Step 3: Use LLM to refine answer
                prompt = f"""
                The user asked: "{query}"
                The database returned: "{raw_answer}"
                Please create a clear and user-friendly answer.
                """
                final_answer = self.__interface_agent(prompt)

            else:
                # Step 4: No answer found
                vector_answer = self.faiss_kb.get_best_answer(query, threshold=0.5)
                print(f"[sql_rag] Vector answer: {vector_answer}")

                if vector_answer == "No similar results found." or vector_answer is None:
                    
                    status = "pending"
                    prompt = f"""
                    The user asked: "{query}"
                    The database could not provide an answer.
                    Write a polite message telling the user 
                    that their question could not be answered 
                    and will be sent to admins for review.
                    """
                    final_answer = self.__interface_agent(prompt)
                else:
                    status = "answered"
                    confidence_score = 0.6
                    prompt = f"""
                    The user asked: "{query}"
                    The database returned: "{vector_answer}"
                    Please create a clear and user-friendly answer.
                    """
                    final_answer = self.__interface_agent(prompt)
                    

        except Exception as e:
            confidence_score = 0.0
            status = "error"
            final_answer = f"An error occurred while processing your request."
            print(f"[sql_rag error] {e}")

        # Step 5: Log the chat into DB
        try:
            self.__chat_logs_model(self.user_id, query, final_answer, confidence_score, status)

        except SQLAlchemyError as e:
            self.session.rollback()
            print(f"[DB log error] {e}")

        return final_answer
    
if __name__ == "__main__":
    crm_sql_module = CRMSQLModule(user_id=1)

    # Test adding a new entry
    inp = ""
    while inp != "exit":
        inp = input("Enter your question (or 'exit' to quit): ")
        print("Answer: \n", crm_sql_module.chat(query=inp))

    # print("\n=== Testing Add Entry ===") 
    # question = "What is the punishment for copying exam?"

    
