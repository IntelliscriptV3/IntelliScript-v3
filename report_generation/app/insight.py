from langchain_openai import ChatOpenAI

_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)

def summarize_dataframe(query: str, df_head: str):
    prompt = f"""
    The user asked: "{query}"
    The top of the SQL result table is:
    {df_head}
    Write a 2–3 sentence summary highlighting trends, totals, or key insights.
    Keep it concise and factual.
    """
    return _llm.invoke(prompt).content.strip()
