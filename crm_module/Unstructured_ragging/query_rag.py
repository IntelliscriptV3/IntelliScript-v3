# query_rag_openai_oo.py
import os
import json
import numpy as np
import faiss
import re
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from config import INDEX_FILE, EMB_FILE, META_FILE, OPENAI_MODEL, EMBED_MODEL, TOP_K,api_key


class RAGQuery:
    """Object-oriented RAG query handler."""

    def __init__(self, openai_api_key=None):
        # OpenAI API setup
        
        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY not set in environment.")
        self.client = OpenAI(api_key=self.api_key)

        # FAISS & embedding placeholders
        self.index = None
        self.embeddings = None
        self.metadata = None
        self.embed_model = SentenceTransformer(EMBED_MODEL)

    # ---------- Load FAISS & Metadata ----------
    def load_resources(self):
        """Load FAISS index, embeddings, and metadata."""
        self.index = faiss.read_index(INDEX_FILE)
        self.embeddings = np.load(EMB_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

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

    # ---------- Generate RAG answer ----------
    def get_rag_answer(self, question, context_chunks, model=OPENAI_MODEL):
        """Return GPT answer using provided context chunks."""
        system_prompt = (
            "You are an assistant that answers questions based ONLY on the provided context. "
            "If the context does not include the answer, say: 'I don't know based on the provided rules.' "
            "Be concise and cite the chunk IDs if possible. "
            "If context has images, acknowledge them as [IMAGE INCLUDED]."
        )

        context_text = "\n\n".join([
            f"Chunk {c['id']}: {c['text']}" + (" [IMAGE INCLUDED]" if c.get("image") else "")
            for c in context_chunks
        ])

        user_prompt = f"Context:\n\n{context_text}\n\nQuestion: {question}\n\nAnswer succinctly and mention which chunks (by ID) support your answer."

        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print("RAG query failed:", e)
            return "Error: Could not get answer."

    # ---------- Full query ----------
    def query(self, question, top_k=TOP_K):
        """Run tagging, FAISS search, and generate RAG answer."""
        # Encode query
        q_emb = self.embed_model.encode([question], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_emb = q_emb.astype('float32')[0]

        # Tagging
        tags = self.get_query_tags(question)
        print("Detected tags:", tags if tags else "[No tags detected]")

        # FAISS search
        context_chunks = self.search_with_tags(q_emb, tags, top_k=top_k)
        print("Using chunks:", [c["id"] for c in context_chunks])

        # Get RAG answer
        answer = self.get_rag_answer(question, context_chunks)
        return answer, context_chunks


# ---------- RUN SCRIPT ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, required=True)
    parser.add_argument("--top-k", "-k", type=int, default=TOP_K)
    args = parser.parse_args()

    rag = RAGQuery()
    answer, chunks = rag.query(args.query, args.top_k)

    print("\n--- RAG ANSWER ---\n")
    print(answer)
    print("\n--- SOURCES ---\n")
    for c in chunks:
        print(f"[Chunk {c['id']}] lines {c.get('lines',[])} tags {c.get('tags',[])}")
        print(c['text'])
        if "image" in c:
            print("[IMAGE INCLUDED]")
        print("----")
