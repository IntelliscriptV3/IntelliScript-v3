# query_rag.py
import json
import numpy as np
import faiss
import re
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
from dotenv import load_dotenv
import os
from .config import INDEX_FILE, EMB_FILE, META_FILE, OPENAI_MODEL, EMBED_MODEL, TOP_K, api_key

load_dotenv()

class RAGQuery:
    """Object-oriented RAG query handler."""

    def __init__(self, user_id: int, openai_api_key=None, db_url=None):
        # --- Setup IDs and Keys ---
        self.user_id = user_id
        self.api_key = openai_api_key or api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY not set in environment.")
        self.client = OpenAI(api_key=self.api_key)

        # --- Setup DB connection ---
        self.db_url = db_url or os.getenv("DB_URL")
        if not self.db_url:
            raise ValueError("❌ DB_URL not found in environment.")
        self.engine = create_engine(self.db_url)
        SessionLocal = sessionmaker(bind=self.engine)
        self.session = SessionLocal()
        self.embed_model = SentenceTransformer(EMBED_MODEL)

        self.index = None
        self.embeddings = None
        self.metadata = None
    # ---------- Load FAISS & Metadata ----------
    def load_resources(self):
        """Load FAISS index, embeddings, and metadata."""
        self.index = faiss.read_index(INDEX_FILE)
        self.embeddings = np.load(EMB_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
    
    def __chat_logs_model(self, query: str, final_answer: str, confidence_score: float, status: str):
        """Insert logs into chat_logs table."""
        insert_query = text("""
            INSERT INTO chat_logs (user_id, question, answer, confidence_score, status, created_at)
            VALUES (:user_id, :question, :answer, :confidence_score, :status, :created_at)
        """)
        try:
            self.session.execute(insert_query, {
                "user_id": self.user_id,
                "question": query,
                "answer": final_answer,
                "confidence_score": confidence_score,
                "status": status,
                "created_at": datetime.now()
            })
            self.session.commit()
        except SQLAlchemyError as e:
            self.session.rollback()
            print(f"[DB log error] {e}")
    # ---------- GPT Tagging ----------
    def get_query_tags(self, query, model=OPENAI_MODEL):
        """Return list of tags for a given query."""
        system_prompt = "You are a concise tagger. Return JSON: {\"tags\":[...]} with up to 3 tags."
        user_prompt = f"User query:\n\n'''{query}'''\n\nReturn tags only as JSON."

        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            content = resp.choices[0].message.content.strip()
            match = re.search(r'\{.*"tags".*\}', content, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return parsed.get("tags", [])
        except Exception as e:
            print("Query tagging failed:", e)
        return []

    # ---------- Search FAISS with optional tag filtering ----------
    def search_with_tags(self, query_emb, tags=None, top_k=TOP_K):
        """
        Returns metadata dicts from FAISS search with optional tag filtering.
        """
        if self.index is None or self.metadata is None or self.embeddings is None:
            self.load_resources()

        if not tags:
            # fallback: search entire index
            D, I = self.index.search(np.array([query_emb]).astype('float32'), top_k)
            return [self.metadata[i] for i in I[0]]

        # Map tags to indices
        tag_to_indices = {}
        for md in self.metadata:
            for t in md.get("tags", []):
                tag_to_indices.setdefault(t.lower(), []).append(md["id"])

        candidate_idxs = set()
        for t in tags:
            candidate_idxs.update(tag_to_indices.get(t.lower(), []))

        if not candidate_idxs:
            # fallback if no chunks with tags found
            D, I = self.index.search(np.array([query_emb]).astype('float32'), top_k)
            return [self.metadata[i] for i in I[0]]

        # Filter embeddings and compute similarity
        cand_list = sorted(list(candidate_idxs))
        cand_embs = self.embeddings[cand_list]
        sims = cand_embs.dot(query_emb.T).flatten()
        top_indices = sims.argsort()[::-1][:top_k]
        return [self.metadata[cand_list[i]] for i in top_indices]

    # ---------- Generate RAG answer with confidence ----------
    def get_rag_answer(self, question, context_chunks, model=OPENAI_MODEL):
        """Return GPT answer + confidence score using provided context chunks."""
        system_prompt = (
            "You are an assistant that answers questions based ONLY on the provided context. "
            "If the context does not include the answer, respond: 'Sending your question to ADMIN.' "
            "Be concise and cite chunk IDs if possible. "
            "After your answer, give a numeric confidence score: 0.8 if data is clearly found and used, 0 if not."
        )

        context_text = "\n\n".join([
            f"Chunk {c['id']}: {c['text']}" + (" [IMAGE INCLUDED]" if c.get('image') else "")
            for c in context_chunks
        ])

        user_prompt = (
            f"Context:\n\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Respond in JSON as:\n"
            "{ \"answer\": \"...\", \"confidence\": <number> }"
        )

        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            content = resp.choices[0].message.content.strip()

            # Try parsing JSON
            match = re.search(r'\{.*"confidence".*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group())
                answer = result.get("answer", "").strip()
                confidence = result.get("confidence", 0)
            else:
                # fallback
                answer = content
                confidence = 0

            return answer, confidence

        except Exception as e:
            print("RAG query failed:", e)
            return "Error: Could not get answer.", 0


    # ---------- Full query ----------
        # ---------- Full query with fallback ----------
    def query(self, question, top_k=TOP_K):
        """Run tagging, FAISS search, generate RAG answer with confidence,
        and fall back to structured FAISSKnowledgeBase if confidence < 0.5.
        """
        from ..structured_ragging.vector_kb import FAISSKnowledgeBase  # Import only when needed

        # --- Step 1: Embed & tag query ---
        q_emb = self.embed_model.encode([question], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_emb = q_emb.astype('float32')[0]

        tags = self.get_query_tags(question)
        print("Detected tags:", tags if tags else "[No tags detected]")

        # --- Step 2: Retrieve from auto-tag FAISS index ---
        context_chunks = self.search_with_tags(q_emb, tags, top_k=top_k)
        print("Using chunks:", [c["id"] for c in context_chunks])

        # --- Step 3: Generate RAG answer ---
        answer, confidence = self.get_rag_answer(question, context_chunks)
        print(f"[Unstructured RAG] Confidence: {confidence:.2f}")

        # --- Step 4: If confidence < 0.5, fallback to structured KB (FAISSKnowledgeBase) ---
        if confidence < 0.5:
            print("[Fallback] Low confidence detected. Searching FAISSKnowledgeBase...")
            kb = FAISSKnowledgeBase()
            kb.load_index()  # load kb_faiss_index.bin and kb_metadata.pkl

            kb_answer = kb.get_best_answer(question, threshold=0.8)
            print("[Fallback Result] KB Answer:", kb_answer)

            if kb_answer == "No similar results found." or kb_answer is None:
                # No answer in structured KB → send to admin
                status = "queued"
                final_answer = "Sending your question to ADMIN."
            else:
                # Found a fallback answer
                status = "answered"
                final_answer = kb_answer
                confidence = 0.6  # moderate confidence for fallback answers
        else:
            status = "answered"
            final_answer = answer

        self.__chat_logs_model(
        query=question,
        final_answer=final_answer,
        confidence_score=confidence,
        status=status
        )
        # --- Step 5: Return full result ---
        return {
            "answer": final_answer,
            "confidence": confidence,
            "status": status,
            "context_chunks": context_chunks
        }



# ---------- RUN SCRIPT ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, required=True)
    parser.add_argument("--top-k", "-k", type=int, default=TOP_K)
    args = parser.parse_args()

    rag = RAGQuery(user_id=1)
    answer, confidence, chunks = rag.query(args.query, args.top_k)


    print("\n--- RAG ANSWER ---\n")
    
