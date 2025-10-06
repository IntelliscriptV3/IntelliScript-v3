import pandas as pd
import plotly.express as px
import plotly.io as pio
import json
import io, base64
import kaleido
from sqlalchemy import text
from sqlalchemy.engine import Engine
from datetime import datetime
pio.kaleido.scope.default_format = "png"
def query_to_df(engine: Engine, sql: str, limit: int = 1000):
    # Clean up any trailing semicolon
    clean_sql = sql.strip().rstrip(";")

    # Only add a limit if not already present
    if "limit" not in clean_sql.lower():
        clean_sql = f"{clean_sql} LIMIT {limit}"

    df = pd.read_sql(text(clean_sql), con=engine)
    return df


def generate_chart(df: pd.DataFrame):
    if df.empty:
        return None, None

    # Heuristic chart rules
    fig = None
    chart_type = None
    cols = df.columns.tolist()

    # 2 columns: one categorical, one numeric → bar or pie
    if len(cols) == 2:
        x, y = cols
        if pd.api.types.is_numeric_dtype(df[y]):
            chart_type = "bar"
            fig = px.bar(df, x=x, y=y, title=f"{y} by {x}")
        else:
            chart_type = "pie"
            fig = px.pie(df, names=x, values=y)
    # Time + numeric → line chart
    elif any("date" in c.lower() or "month" in c.lower() for c in cols):
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            chart_type = "line"
            fig = px.line(df, x=cols[0], y=num_cols, title="Trends Over Time")
    # Fallback: show table only
    if fig is None:
        return None, None

    img_bytes = pio.to_image(fig, format="png")
    img_b64 = base64.b64encode(img_bytes).decode()
    return fig, img_b64

def dataframe_to_html(df: pd.DataFrame, max_rows=20):
    return df.head(max_rows).to_html(index=False, classes="table table-striped")

def process_sql_result(engine, sql):
    df = query_to_df(engine, sql)
    fig, img_b64 = generate_chart(df)
    html_table = dataframe_to_html(df)
    return {"df": df, "chart_img": img_b64, "html_table": html_table}





#Phase 01
# import re
# import pandas as pd
# import plotly.express as px
# from sqlalchemy import text
# from sqlalchemy.engine import Engine
# import json

# def _extract_sql_from_agent_text(agent_text: str) -> str | None:
#     # Very simple heuristic: look for a SQL block; refine later with tool metadata
#     m = re.search(r"(?is)```sql\s*(.+?)```", agent_text)
#     return m.group(1).strip() if m else None

# def auto_chart_for_result(engine: Engine, query: str, agent_text: str):
#     """
#     If the agent_text includes a SQL block and result is small and structured, render a chart.
#     """
#     narrative = agent_text
#     figs_json = []

#     sql = _extract_sql_from_agent_text(agent_text)
#     if not sql:
#         return {"narrative": narrative, "figs_json": figs_json}

#     # Ensure it's SELECT only
#     dangerous = ["update ", "delete ", "insert ", "alter ", "drop ", "create ", "truncate "]
#     if any(tok in sql.lower() for tok in dangerous):
#         return {"narrative": narrative, "figs_json": figs_json}

#     # Query DB
#     df = pd.read_sql(text(sql), con=engine)
#     if df.empty or len(df.columns) == 0:
#         return {"narrative": narrative, "figs_json": figs_json}

#     # Simple auto chart rules
#     fig = None
#     if df.shape[1] == 2:
#         # If one categorical and one numeric -> bar
#         c0, c1 = df.columns
#         if pd.api.types.is_numeric_dtype(df[c1]):
#             fig = px.bar(df, x=c0, y=c1, title=f"{c1} by {c0}")
#     elif df.shape[1] >= 2 and all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns[1:]):
#         # Multiple numeric columns -> line vs first column if non-numeric
#         xcol = df.columns[0]
#         if not pd.api.types.is_numeric_dtype(df[xcol]):
#             long_df = df.melt(id_vars=[xcol], var_name="metric", value_name="value")
#             fig = px.line(long_df, x=xcol, y="value", color="metric", title="Trends")

#     if fig is not None:
#         figs_json.append(json.loads(fig.to_json()))  # Streamlit/React can render from JSON

#     return {"narrative": narrative, "figs_json": figs_json}
