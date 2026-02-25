# database.py
import chromadb
import sys
from config import DB_PATH

class VectorDB:
    def __init__(self):
        print("⏳ Connecting to ChromaDB...")
        try:
            self.client = chromadb.PersistentClient(path=DB_PATH)
            self.collection = self.client.get_or_create_collection(
                name="face_embeddings",
                metadata={"hnsw:space": "cosine"} 
            )
            print(f"✅ ChromaDB Connected at {DB_PATH}")
        except Exception as e:
            print(f"❌ ChromaDB Error: {e}")
            sys.exit(1)

    def search(self, embedding, threshold):
        if self.collection.count() == 0:
            return None, None
        
        # FIX 3: Pull the top 3 closest matches instead of just 1
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=3
        )
        
        if results['distances'] and len(results['distances'][0]) > 0:
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]
            
            # Loop through the top 3 to see if ANY of them are a match
            for i in range(len(distances)):
                if distances[i] < threshold:es[i]
            
            # If we checked all 3 and none were below threshold, return None
            return None, distances[0] # Return closest distance for debugging
            
        return None, None

    def add_record(self, record_id, embedding, metadata):
        self.collection.add(
            ids=[record_id], 
            embeddings=[embedding],
            metadatas=metadata
        )