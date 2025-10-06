import os, json
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

from app.db import engine, SessionLocal
from app.eval import evaluate_confidence
#from app.charts import auto_chart_for_result
from app.charts import process_sql_result
from app.insight import summarize_dataframe

LAST_SQL_QUERY = None

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # align with your friend’s code

# Only expose tables we want the agent to touch (safer prompts & better SQL)
ALLOWED_TABLES = [
    "users","students","teachers","courses","enrollments",
    "attendance","fees","assessments","assessment_results",
    "chat_logs","knowledge_base"
]

def build_agent():
    db = SQLDatabase.from_uri(
        os.getenv("DB_URL"),
        include_tables=ALLOWED_TABLES,
        sample_rows_in_table_info=3,
        view_support=False
    )

    # --- Capture every query executed by the SQL agent ---
    global LAST_SQL_QUERY
    LAST_SQL_QUERY = None

    original_run = db.run

    def capture_run(query: str, *args, **kwargs):
        """Intercept every SQL query executed by the LLM agent."""
        global LAST_SQL_QUERY
        LAST_SQL_QUERY = query
        # Pass through all arguments to the original run()
        return original_run(query, *args, **kwargs)

    db.run = capture_run

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    agent = create_sql_agent(
        llm,
        db=db,
        agent_type="tool-calling",
        verbose=True,
        return_intermediate_steps=True   # ✅ correct flag name
    )
    return agent


_agent = build_agent()

SYSTEM_SAFETY = """You are a reporting assistant. Use ONLY SELECT queries.
Never update/insert/delete/alter data. If a user asks for a mutation, refuse.
Prefer concise SQL. If query results are large, aggregate (COUNT/SUM/AVG)."""

def chat(user_id: int, query: str) -> dict:
    session = SessionLocal()
    status = "failed"
    final_answer = ""
    confidence_score = 0.0
    figs_json = []

    try:
        # Ask the SQL agent
        response = _agent.invoke({"input": f"{SYSTEM_SAFETY}\nUser: {query}"})
        raw_answer = response.get("output", "")

        # Confidence gating via LLM-JSON (and simple heuristics inside)
        confidence_score, reason = evaluate_confidence(query, raw_answer)

        # If low confidence → push to admin queue and craft a polite message
        if confidence_score < CONFIDENCE_THRESHOLD:
            status = "pending"
            final_answer = (
                "I couldn’t confidently answer this from the database. "
                "I’ve sent your question to an admin for review."
            )
        else:
            # --- Phase 2: visualization + summary ---
            status = "answered"
            sql_block = None
            import re

            # 1️⃣ Try extracting SQL from fenced block in LLM output
            m = re.search(r"(?is)```sql\s*(.+?)```", raw_answer)
            sql_block = m.group(1).strip() if m else None

            # 2️⃣ Try extracting SQL from agent internal steps (if available)
            if not sql_block and "intermediate_steps" in response:
                for step in response["intermediate_steps"]:
                    # LangChain tool usually called 'sql_db_query'
                    if isinstance(step, tuple) and "sql_db_query" in str(step[0]).lower():
                        step_data = step[1]
                        if isinstance(step_data, dict) and "query" in step_data:
                            sql_block = step_data["query"]
                            break

            # 3️⃣ Fallback: detect any SELECT manually from text
            if not sql_block:
                alt = re.search(r"(?is)(SELECT\s.+?)(?:;|\n|$)", raw_answer)
                if alt:
                    sql_block = alt.group(1).strip()

            from app.agent import LAST_SQL_QUERY

            # Use the captured SQL if regex and intermediate_steps failed
            if not sql_block and LAST_SQL_QUERY:
                sql_block = LAST_SQL_QUERY

            print("🧠 Detected SQL:", sql_block)


            # 4️⃣ Visualization & Summary
            if sql_block:
                

                try:
                    df = pd.read_sql(text(sql_block), con=engine)
                    visuals = process_sql_result(engine, sql_block)
                    summary = summarize_dataframe(query, df.head().to_string())

                    final_answer = (
                        f"### 📊 Report Summary\n{summary}\n\n---\n\n{visuals['html_table']}"
                    )

                    if visuals["chart_img"]:
                        final_answer += f"\n\n![chart](data:image/png;base64,{visuals['chart_img']})"

                except Exception as vis_err:
                    print("Visualization error:", vis_err)
                    final_answer = f"Database response:\n{raw_answer}\n\n(Chart rendering failed.)"

            else:
                # No SQL found at all → fallback
                final_answer = f"Database response:\n{raw_answer}\n\n"

            # status = "answered"
            # # Try auto charts if a table is present in the raw answer (best-effort)
            # chart_res = auto_chart_for_result(
            #     engine=engine, query=query, agent_text=raw_answer
            # )
            # final_answer = chart_res["narrative"]
            # figs_json = chart_res["figs_json"]  # list of Plotly figure JSON

        # Log to chat_logs
        try:
            insert = text("""
                INSERT INTO chat_logs (user_id, question, answer, confidence_score, status, created_at)
                VALUES (:user_id, :q, :a, :c, :s, :ts)
                RETURNING chat_id
            """)
            chat_id = session.execute(insert, {
                "user_id": user_id, "q": query, "a": final_answer,
                "c": float(confidence_score), "s": status, "ts": datetime.now()
            }).scalar()
            session.commit()

            # If pending, enqueue for admin
            if status == "pending":
                # If your table is adminQueue, either create a view or rename accordingly
                try:
                    session.execute(text("""
                        INSERT INTO admin_queue (chat_id, admin_id)
                        VALUES (:cid, NULL)
                    """), {"cid": chat_id})
                    session.commit()
                except SQLAlchemyError:
                    session.rollback()  # if admin_queue doesn’t exist yet, just skip for now

        except SQLAlchemyError:
            session.rollback()

        return {
            "status": status,
            "answer": final_answer,
            "confidence": confidence_score,
            "figures": figs_json,   # list of Plotly figure dicts (for UI)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "answer": f"An internal error occurred: {str(e)}",
            "confidence": 0.0,
            "figures": []
        }

    finally:
        session.close()
