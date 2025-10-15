import os
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, create_engine, text
from sqlalchemy.exc import SQLAlchemyError


class AdminQueueGeneration:
    def __init__(self, db_link="postgresql://postgres:5737@localhost:5433/intelliscript2"):
        self.engine = create_engine(db_link)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def transfer_pending_to_admin(self):
        """
        Check chat_logs table for rows with status='pending' and transfer their questions to admin_queue table.
        """
        session = self.SessionLocal()
        try:
            # Step 1: Get all pending rows from chat_logs
            select_pending_query = text("""
                SELECT chat_id, user_id, question, created_at 
                FROM chat_logs 
                WHERE status = 'pending'
                ORDER BY created_at ASC
            """)
            
            pending_rows = session.execute(select_pending_query).fetchall()
            
            if not pending_rows:
                print("No pending questions found in chat_logs.")
                return {"transferred": 0, "message": "No pending questions found"}
            
            transferred_count = 0
            
            # Step 2: Process each pending row
            for row in pending_rows:
                chat_log_id = row[0]
                user_id = row[1]
                question = row[2]
                # original_created_at = row[3]
                
                try:
                    # Step 3: Insert question into admin_queue
                    insert_admin_query = text("""
                        INSERT INTO admin_queue (chat_id, question, assigned_at)
                        VALUES (:chat_id, :question, :assigned_at)
                    """)
                    
                    session.execute(insert_admin_query, {
                        "chat_id": chat_log_id,
                        "question": question,
                        "assigned_at": datetime.now()
                    })
                    
                    # Step 4: Update chat_logs status to 'transferred_to_admin'
                    update_chat_log_query = text("""
                        UPDATE chat_logs 
                        SET status = 'queued'
                        WHERE chat_id = :chat_log_id
                    """)
                    
                    session.execute(update_chat_log_query, {
                        "chat_log_id": chat_log_id
                    
                    })
                    
                    transferred_count += 1
                    print(f"Transferred question from chat_log ID {chat_log_id} to admin_queue")
                    
                except SQLAlchemyError as e:
                    print(f"Error transferring chat_log ID {chat_log_id}: {e}")
                    session.rollback()
                    raise e
            
            # Step 5: Commit all changes
            session.commit()
            
            print(f"Successfully transferred {transferred_count} pending questions to admin_queue.")
            return {
                "transferred": transferred_count, 
                "message": f"Successfully transferred {transferred_count} pending questions to admin_queue"
            }
            
        except SQLAlchemyError as e:
            session.rollback()
            print(f"Database error: {e}")
            return {"error": str(e), "transferred": 0}
            
        except Exception as e:
            session.rollback()
            print(f"Unexpected error: {e}")
            return {"error": str(e), "transferred": 0}
            
        finally:
            session.close()

    def get_pending_questions_count(self):
        """
        Get count of pending questions in chat_logs table.
        """
        session = self.SessionLocal()
        
        try:
            count_query = text("""
                SELECT COUNT(*) 
                FROM chat_logs 
                WHERE status = 'pending'
            """)
            
            result = session.execute(count_query).scalar()
            return result
            
        except SQLAlchemyError as e:
            print(f"Error getting pending count: {e}")
            return 0
            
        finally:
            session.close()


    # Main execution function
    def generate(self):
        """
        Main function to execute the transfer process with summary information.
        """
        print("=== Pending Questions Transfer Process ===")
        
        # Check current pending count
        pending_count = self.get_pending_questions_count()
        print(f"Found {pending_count} pending questions in chat_logs")
        
        if pending_count == 0:
            print("No pending questions to transfer.")
            return
        
        print("\nCurrent admin_queue summary:")      
        
        # Perform the transfer
        print(f"\nTransferring {pending_count} pending questions...")
        result = self.transfer_pending_to_admin()
        
        if "error" in result:
            print(f"Transfer failed: {result['error']}")
        else:
            print(f"Transfer completed: {result['message']}")
            
        

if __name__ == "__main__":
    
    generator = AdminQueueGeneration()
    generator.generate()