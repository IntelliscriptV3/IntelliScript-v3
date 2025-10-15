import os, requests, json
import streamlit as st
import plotly.graph_objs as go

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")

st.set_page_config(page_title="IntelliScript Reports", layout="wide")
st.title("IntelliScript – SQL Reports Assistant")

if "history" not in st.session_state:
    st.session_state["history"] = []

def render_fig(fig_dict):
    fig = go.Figure(fig_dict)
    st.plotly_chart(fig, use_container_width=True)

user_input = st.chat_input("Ask about students, attendance, fees, assessments...")

if user_input:
    payload = {"user_id": 1, "message": user_input}
    resp = requests.post(API_URL, json=payload, timeout=90).json()
    st.session_state["history"].append({"role":"user","content":user_input})
    st.session_state["history"].append({"role":"assistant","content":resp["answer"],"figs":resp.get("figures",[]),"confidence":resp.get("confidence")})

for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        if msg["role"]=="assistant":
            st.write(f"**Confidence:** {msg.get('confidence',0):.2f}")
            st.markdown(msg["content"], unsafe_allow_html=True)

            for f in msg.get("figs", []):
                render_fig(f)
        else:
            st.markdown(msg["content"], unsafe_allow_html=True)

