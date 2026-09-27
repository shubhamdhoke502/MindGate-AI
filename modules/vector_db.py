import os
import zipfile
import shutil
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

class VectorStoreManager:
    def __init__(self, db_dir: str = "./storage/chroma_db", model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initializes ChromaDB with a local SentenceTransformer embedding model.
        """
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Initialize Local Embedding Model
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        
        # Initialize Local ChromaDB Persistent Client
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(
            name="mindgate_documents",
            embedding_function=self.embedding_fn
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Inserts parsed chunks with metadata coordinates into ChromaDB.
        """
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["content_payload"]["raw_text"] for c in chunks]
        
        metadatas = []
        for c in chunks:
            metadatas.append({
                "file_name": c["document_metadata"]["file_name"],
                "page_number": c["chunk_coordinates"]["page_number"],
                "line_start": c["chunk_coordinates"]["line_start"],
                "line_end": c["chunk_coordinates"]["line_end"],
                "character_count": c["chunk_coordinates"]["character_count"]
            })
            
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def query_similar_chunks(self, query: str, top_k: int = 2, file_name_filter: str = None) -> List[Dict[str, Any]]:
        """
        Searches for top_k relevant chunks. Supports exact file_name filtering.
        """
        where_filter = {"file_name": file_name_filter} if file_name_filter else None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )
        
        retrieved_items = []
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                retrieved_items.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results and results["distances"] else None
                })
                
        return retrieved_items


class StorageCompressor:
    @staticmethod
    def compress_workspace(source_dir: str, output_zip_path: str) -> str:
        """
        Compresses workspace/ChromaDB files to save up to 80% disk space (.mgdb file).
        """
        base_name = output_zip_path.replace('.zip', '').replace('.mgdb', '')
        archive_path = shutil.make_archive(base_name, 'zip', source_dir)
        final_path = base_name + ".mgdb"
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(archive_path, final_path)
        return final_path

    @staticmethod
    def decompress_workspace(compressed_path: str, extract_to_dir: str) -> None:
        """
        Decompresses .mgdb workspace back to disk on-demand.
        """
        os.makedirs(extract_to_dir, exist_ok=True)
        with zipfile.ZipFile(compressed_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_dir)


# Quick Test Run
if __name__ == "__main__":
    from parser import parse_pdf_with_coordinates
    
    # Test Vector Pipeline
    v_store = VectorStoreManager()
    print("Vector Store Initialized Successfully.")