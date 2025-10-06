import os
import pickle
import numpy as np
import faiss
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sentence_transformers import SentenceTransformer
import json

# --- Database Setup ---
engine = create_engine("postgresql://postgres:kisal123@localhost:5432/intelliscript2")
SessionLocal = sessionmaker(bind=engine)

class FAISSKnowledgeBase:
    def __init__(self, model_name='all-MiniLM-L6-v2', index_file='kb_faiss_index.bin', metadata_file='kb_metadata.pkl'):
        """
        Initialize FAISS Knowledge Base
        
        Args:
            model_name: Sentence transformer model name for embeddings
            index_file: File to save/load FAISS index
            metadata_file: File to save/load question-answer metadata
        """
        self.model_name = model_name
        self.index_file = index_file
        self.metadata_file = metadata_file
        
        # Initialize sentence transformer model
        print(f"Loading sentence transformer model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        
        # Initialize FAISS index and metadata
        self.index = None
        self.metadata = []
        self.dimension = None
        
    def fetch_knowledge_base_data(self):
        """
        Fetch all questions and answers from knowledge_base table
        
        Returns:
            List of dictionaries containing kb_id, question, answer, answered_by, created_at
        """
        session = SessionLocal()
        
        try:
            query = text("""
                SELECT kb_id, question, answer, answered_by, created_at
                FROM knowledge_base
                WHERE question IS NOT NULL AND answer IS NOT NULL
                ORDER BY created_at ASC
            """)
            
            results = session.execute(query).fetchall()
            
            kb_data = []
            for row in results:
                kb_data.append({
                    'kb_id': row[0],
                    'question': row[1],
                    'answer': row[2],
                    'answered_by': row[3],
                    'created_at': row[4]
                })
            
            print(f"Fetched {len(kb_data)} question-answer pairs from knowledge base")
            return kb_data
            
        except SQLAlchemyError as e:
            print(f"Error fetching knowledge base data: {e}")
            return []
            
        finally:
            session.close()
    
    def create_embeddings(self, texts):
        """
        Create embeddings for a list of texts using sentence transformer
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of embeddings
        """
        print(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_faiss_index(self, save_index=True):
        """
        Build FAISS index from knowledge base data
        
        Args:
            save_index: Whether to save the index and metadata to files
            
        Returns:
            Tuple of (faiss_index, metadata)
        """
        print("Building FAISS index from knowledge base...")
        
        # Fetch data from database
        kb_data = self.fetch_knowledge_base_data()
        
        if not kb_data:
            print("No knowledge base data found!")
            return None, None
        
        # Prepare texts for embedding (combine question and answer for better search)
        texts = []
        metadata = []
        
        for item in kb_data:
            # Combine question and answer for richer embedding
            combined_text = f"Question: {item['question']} Answer: {item['answer']}"
            texts.append(combined_text)
            
            # Store metadata
            metadata.append({
                'kb_id': item['kb_id'],
                'question': item['question'],
                'answer': item['answer'],
                'answered_by': item['answered_by'],
                'created_at': item['created_at'].isoformat() if item['created_at'] else None,
                'combined_text': combined_text
            })
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Initialize FAISS index
        self.dimension = embeddings.shape[1]
        print(f"Embedding dimension: {self.dimension}")
        
        # Create FAISS index (using IndexFlatIP for cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        self.metadata = metadata
        
        print(f"FAISS index built successfully with {self.index.ntotal} vectors")
        
        # Save index and metadata if requested
        if save_index:
            self.save_index()
        
        return self.index, self.metadata
    
    def save_index(self):
        """Save FAISS index and metadata to files"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)
            print(f"FAISS index saved to {self.index_file}")
            
            # Save metadata
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            print(f"Metadata saved to {self.metadata_file}")
            
            # Save additional info
            info = {
                'model_name': self.model_name,
                'dimension': self.dimension,
                'total_vectors': self.index.ntotal,
                'created_at': datetime.now().isoformat()
            }
            
            with open('kb_index_info.json', 'w') as f:
                json.dump(info, f, indent=2)
            print("Index info saved to kb_index_info.json")
            
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def load_index(self):
        """Load FAISS index and metadata from files"""
        try:
            # Load FAISS index
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
                print(f"FAISS index loaded from {self.index_file}")
            else:
                print(f"Index file {self.index_file} not found")
                return False
            
            # Load metadata
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"Metadata loaded from {self.metadata_file}")
            else:
                print(f"Metadata file {self.metadata_file} not found")
                return False
            
            self.dimension = self.index.d
            print(f"Loaded index with {self.index.ntotal} vectors, dimension: {self.dimension}")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def search_similar(self, query, k=5, threshold=0.5):
        """
        Search for similar questions/answers in the knowledge base
        
        Args:
            query: Search query string
            k: Number of top results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of search results with similarity scores
        """
        if self.index is None:
            print("Index not loaded. Please build or load the index first.")
            return []
        
        # Create embedding for query
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if score >= threshold:  # Filter by threshold
                result = {
                    'rank': i + 1,
                    'similarity_score': float(score),
                    'kb_id': self.metadata[idx]['kb_id'],
                    'question': self.metadata[idx]['question'],
                    'answer': self.metadata[idx]['answer'],
                    'answered_by': self.metadata[idx]['answered_by'],
                    'created_at': self.metadata[idx]['created_at']
                }
                results.append(result)
        
        return results
    
    def get_best_answer(self, query, threshold=0.5):
        """
        Get the highest similarity answer for a query
        
        Args:
            query: Search query string
            threshold: Minimum similarity score (0-1)
            
        Returns:
            Dictionary with the best match or None if no match above threshold
        """
        results = self.search_similar(query, k=1, threshold=threshold)
        return results[0]['answer'] if results else "No similar results found."
    
    def get_index_stats(self):
        """Get statistics about the current index"""
        if self.index is None:
            return {"status": "No index loaded"}
        
        stats = {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "model_name": self.model_name,
            "index_file": self.index_file,
            "metadata_file": self.metadata_file
        }
        
        return stats
    
    def rebuild_index(self):
        """Rebuild the entire index from current database"""
        print("Rebuilding FAISS index from current knowledge base...")
        return self.build_faiss_index(save_index=True)

def main():
    """Main function to demonstrate FAISS knowledge base creation"""
    print("=== FAISS Knowledge Base Builder ===")
    
    # Initialize FAISS KB
    faiss_kb = FAISSKnowledgeBase()
    
    # Check if index already exists
    if os.path.exists(faiss_kb.index_file) and os.path.exists(faiss_kb.metadata_file):
        print("\nExisting index found. Choose an option:")
        print("1. Load existing index")
        print("2. Rebuild index from database")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            success = faiss_kb.load_index()
            if not success:
                print("Failed to load existing index. Building new index...")
                faiss_kb.build_faiss_index()
        else:
            faiss_kb.rebuild_index()
    else:
        # Build new index
        faiss_kb.build_faiss_index()
    
    # Display index statistics
    stats = faiss_kb.get_index_stats()
    print(f"\nIndex Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test search functionality
    print("\n=== Testing Search Functionality ===")
    test_query = "What happens if I copy exams?"
    print(f"Searching for: '{test_query}'")
    
    results = faiss_kb.get_best_answer(test_query, threshold=0.1)
    print(f"Best Answer: {results}")
    # if results:
    #     best_result = results[0]  # Get the highest similar result
    #     print(f"\nBest match (Score: {best_result['similarity_score']:.3f}):")
    #     print(f"Question: {best_result['question']}")
    #     print(f"Answer: {best_result['answer']}")
    # else:

    return faiss_kb

if __name__ == "__main__":
    faiss_kb = main()
