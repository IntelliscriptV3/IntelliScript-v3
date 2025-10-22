import os, json, re
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import sqlparse

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

from .db import engine, SessionLocal
from .eval import evaluate_confidence
from .charts import process_sql_result
from .insight import summarize_dataframe

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


# -------------------------------------------------
# 🧠 Role-Aware System Prompt
# -------------------------------------------------
def get_system_safety(user_role: str, user_id: int) -> str:
    """
    Returns a role-aware safety instruction to the LLM.
    """
    base_prompt = """
You are a reporting assistant. Use ONLY SELECT queries.
Never update, insert, delete, or alter any data.
Prefer concise SQL. If query results are large, aggregate using COUNT/SUM/AVG.
"""

    if user_role == "student":
        base_prompt += f"""
You are talking to a student with student_id={user_id}.
Only show details that belong to this student.
Never include or summarize data from other students.
"""
    elif user_role == "admin":
        base_prompt += "You are talking to an admin. You can access all student data.\n"

    return base_prompt.strip()

# -------------------------------------------------
# 🔒 Role-Based Access Control (RBAC)
# -------------------------------------------------
def apply_rbac(sql_block: str, user_role: str, user_id: int) -> str:
    """
    Safely enforces role-based access control by modifying SQL queries.
    Ensures students only see their own data.
    """

    if user_role == "admin":
        return sql_block  # no restriction for admins

    if user_role != "student" or not sql_block:
        return sql_block

    parsed = sqlparse.parse(sql_block)
    if not parsed:
        return sql_block

    stmt = parsed[0]
    sql_lower = sql_block.lower()

    # Only modify queries that touch student-related tables
    restricted_tables = ["students", "attendance", "fees", "assessment_results", "enrollments"]

    # If no restricted tables appear, skip filtering
    if not any(tbl in sql_lower for tbl in restricted_tables):
        return sql_block

    # Add the student_id filter safely
    if "where" in sql_lower:
        enforced_sql = re.sub(
            r"\bwhere\b", f"WHERE student_id = {user_id} AND ", sql_block, flags=re.IGNORECASE
        )
    else:
        enforced_sql = sql_block.rstrip(";") + f" WHERE student_id = {user_id};"

    print("🧱 RBAC-enforced SQL:", enforced_sql)
    return enforced_sql


def chat(user_id: int, query: str, user_role: str = "student") -> dict:
    session = SessionLocal()
    status = "failed"
    final_answer = ""
    confidence_score = 0.0
    figs_json = []

    try:
        # --- 1️⃣ Ask the SQL Agent ---
        system_prompt = get_system_safety(user_role, user_id)
        response = _agent.invoke({"input": f"{system_prompt}\nUser: {query}"})
        raw_answer = response.get("output", "")
        print("🧩 LLM raw answer:", raw_answer)

        # --- 2️⃣ Confidence Check ---
        confidence_score, reason = evaluate_confidence(query, raw_answer)

        if confidence_score < CONFIDENCE_THRESHOLD:
            status = "pending"
            final_answer = (
                "I couldn’t confidently answer this from the database. "
                "I’ve sent your question to an admin for review."
            )
        else:
            status = "answered"

            # --- 3️⃣ Extract SQL from LLM Output ---
            sql_block = None
            m = re.search(r"(?is)```sql\s*(.+?)```", raw_answer)
            if m:
                sql_block = m.group(1).strip()

            if not sql_block and "intermediate_steps" in response:
                for step in response["intermediate_steps"]:
                    if isinstance(step, tuple) and "sql_db_query" in str(step[0]).lower():
                        data = step[1]
                        if isinstance(data, dict) and "query" in data:
                            sql_block = data["query"]
                            break

            if not sql_block:
                alt = re.search(r"(?is)(SELECT\s.+?)(?:;|\n|$)", raw_answer)
                if alt:
                    sql_block = alt.group(1).strip()

            global LAST_SQL_QUERY
            if not sql_block and LAST_SQL_QUERY:
                sql_block = LAST_SQL_QUERY

            print("🧠 Detected SQL:", sql_block)

            # --- 4️⃣ Apply Role-Based Filtering ---
            if sql_block:
                sql_block = apply_rbac(sql_block, user_role, user_id)

                try:
                    df = pd.read_sql(text(sql_block), con=engine)
                    visuals = process_sql_result(engine, sql_block)
                    summary = summarize_dataframe(query, df.head().to_string())

                    final_answer = (
                        f"### 📊 Report Summary\n{summary}\n\n---\n\n{visuals['html_table']}"
                    )

                    if visuals.get("chart_img"):
                        final_answer += f"\n\n![chart](data:image/png;base64,{visuals['chart_img']})"

                except Exception as vis_err:
                    print("Visualization error:", vis_err)
                    final_answer = f"Database response:\n{raw_answer}\n\n(Chart rendering failed.)"
            else:
                final_answer = f"Database response:\n{raw_answer}\n\n"

        # --- 5️⃣ Log to Database ---
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

            # Queue for admin if low confidence
            if status == "pending":
                try:
                    session.execute(text("""
                        INSERT INTO admin_queue (chat_id, admin_id)
                        VALUES (:cid, NULL)
                    """), {"cid": chat_id})
                    session.commit()
                except SQLAlchemyError:
                    session.rollback()

        except SQLAlchemyError:
            session.rollback()

        return {
            "status": status,
            "answer": final_answer,
            "confidence": confidence_score,
            "figures": figs_json,
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
