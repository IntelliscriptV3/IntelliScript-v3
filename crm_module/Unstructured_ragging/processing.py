# processing.py
import os
import json
import glob
from tqdm import tqdm
import numpy as np
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from document_readers import ReaderFactory
from config import GROUP_SIZE, TOP_TAGS, INDEX_FILE, EMB_FILE, META_FILE, OPENAI_MODEL, EMBED_MODEL, api_key

class Chunker:
    @staticmethod
    def text_chunks(lines, group_size=GROUP_SIZE):
        chunks = []
        for i in range(0, len(lines), group_size):
            grp = lines[i:i + group_size]
            text = "\n".join(grp)
            chunks.append({"text": text, "lines": list(range(i + 1, i + 1 + len(grp)))})
        return chunks

    @staticmethod
    def table_chunks(tables):
        chunks = []
        for t in tables:
            for row in t:
                text = " | ".join(str(cell) for cell in row if str(cell).strip())
                if text:
                    chunks.append({"text": text, "lines": []})
        return chunks

    @staticmethod
    def image_chunks(images):
        return [{"text": f"[Image {i+1}]", "lines": [], "image": img} for i, img in enumerate(images)]

class Tagger:
    def __init__(self, model=OPENAI_MODEL):
        
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def tag(self, text, max_tags=TOP_TAGS):
        system_prompt = (
            f"You are a terse tagger. Given the input text, return JSON with up to {max_tags} short tags (1-3 words). Only output valid JSON."
        )
        user_prompt = f"Text:\n'''{text}'''\n\nReturn tags as JSON like: {{\"tags\":[\"HR\",\"Attendance\"]}}"
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}]
            )
            content = resp.choices[0].message.content.strip()
            json_start = content.rfind('{')
            json_end = content.rfind('}') + 1
            parsed = json.loads(content[json_start:json_end])
            tags = parsed.get("tags", [])
            return [t.strip() for t in tags if t.strip()]
        except Exception as e:
            print("Tagging failed:", e)
            return []

class Embedder:
    def __init__(self, model_name=EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def encode(self, chunks):
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings

class RAGPipeline:
    def __init__(self, source_folder):
        self.source_folder = source_folder
        self.chunker = Chunker()
        self.tagger = Tagger()
        self.embedder = Embedder()

    def process_files(self):
        all_chunks = []
        files = glob.glob(os.path.join(self.source_folder, "*.*"))
        print(f"Found {len(files)} files in {self.source_folder}")

        for file_path in files:
            ext = file_path.split(".")[-1]
            reader = ReaderFactory.get_reader(ext)
            if not reader:
                print(f"⚠️ Skipping unsupported file type: {file_path}")
                continue
            print(f"📄 Processing {file_path} ...")
            lines, tables, images = reader.read(file_path)
            chunks = (
                self.chunker.text_chunks(lines) +
                self.chunker.table_chunks(tables) +
                self.chunker.image_chunks(images)
            )
            all_chunks.extend(chunks)
        return all_chunks

    def tag_chunks(self, chunks):
        print("🔖 Tagging chunks...")
        for i, chunk in enumerate(tqdm(chunks)):
            chunk["tags"] = self.tagger.tag(chunk["text"])
            if i < 3:
                print(f"\nChunk {i+1}: {chunk['text']}\nTags: {chunk['tags']}")
        return chunks

    def build_index(self, chunks):
        print("🔍 Building embeddings and FAISS index...")
        embeddings = self.embedder.encode(chunks)
        faiss.normalize_L2(embeddings)
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings.astype("float32"))
        faiss.write_index(index, INDEX_FILE)
        np.save(EMB_FILE, embeddings.astype("float32"))

        metadata = []
        for i, c in enumerate(chunks):
            meta = {
                "id": i,
                "text": c["text"],
                "tags": c.get("tags", []),
                "lines": c.get("lines", [])
            }
            if "image" in c:
                meta["image"] = c["image"]
            metadata.append(meta)

        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved index -> {INDEX_FILE}, embeddings -> {EMB_FILE}, metadata -> {META_FILE}")
