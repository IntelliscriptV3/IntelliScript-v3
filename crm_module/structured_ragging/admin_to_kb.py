from sqlalchemy import Column, create_engine, text, Integer, Text, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime


class KBGeneration:
    def __init__(self, admin_id: int, db_link="postgresql://postgres:5737@localhost:5433/intelliscript2"):
        self.engine = create_engine(db_link)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.admin_id = admin_id

    def load_unanswered_questions(self):
        """Load unanswered questions from admin_queue"""
        session = self.SessionLocal()
        
        try:
            # Query for questions where answer is null
            query = text("""
                SELECT queue_id, chat_id, question, assigned_at
                FROM admin_queue 
                WHERE answer IS NULL
                ORDER BY assigned_at ASC
            """)
            
            results = session.execute(query).fetchall()           

            unanswered_questions = []
            
            # Add questions to list
            for row in results:
                queue_id, chat_id, question, assigned_at = row
            
                unanswered_questions.append({
                    'queue_id': queue_id,
                    'chat_id': chat_id,
                    'question': question,
                    'assigned_at': assigned_at
                })
            
            return unanswered_questions
            
        except SQLAlchemyError as e:
            print(f"Error loading unanswered questions: {str(e)}")            
            return []

        finally:
            session.close()

    def submit_answer(self, queue_id: int, answer: str):

        session = self.SessionLocal()

        try:
            # Get the question details
            get_question_query = text("""
                SELECT question FROM admin_queue 
                WHERE queue_id = :queue_id
            """)
            
            result = session.execute(get_question_query, {"queue_id": queue_id}).fetchone()
            
            if not result:
                print("Error: Question not found.")
                return
            
            question = result[0]
            
            # Insert into knowledge_base
            insert_kb_query = text("""
                INSERT INTO knowledge_base (question, answer, answered_by, created_at)
                VALUES (:question, :answer, :answered_by, :created_at)
            """)
            
            session.execute(insert_kb_query, {
                "question": question,
                "answer": answer,
                "answered_by": self.admin_id,
                "created_at": datetime.now()
            })
            
            # Update admin_queue with answer
            update_admin_query = text("""
                UPDATE admin_queue 
                SET answer = :answer, answered_by = :answered_by
                WHERE queue_id = :queue_id
            """)
            
            session.execute(update_admin_query, {
                "answer": answer,
                "answered_by": self.admin_id,
                "queue_id": queue_id
            })
            
            session.commit()

            print("Success: Answer submitted successfully and added to knowledge base!")

        
        except SQLAlchemyError as e:
            session.rollback()
            print(f"Database error: {str(e)}")
                        
        finally:
            session.close()

    def filling(self):
        """Function to fill unanswered questions from admin_queue"""
        unanswered_qs = self.load_unanswered_questions()

        try:
            if len(unanswered_qs) == 0:
                print("No unanswered questions in admin_queue.")
                return
            
            for q in unanswered_qs:
                print(q['question'])
                ans = input("Your answer: ")
                self.submit_answer(queue_id=q['queue_id'], answer=ans)
                print("-----")

        except Exception as e:
            print(f"Error occurred: {str(e)}")

        print("All done!")

if __name__ == "__main__":
    
    generator = KBGeneration(admin_id=1)
    generator.filling()