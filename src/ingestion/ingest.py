from utils.utils import *
import os
import re
import chromadb
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore

class ChromaIngester():
    def __init__(self, db_file_path, collection_name):
        self.logger = Logger("chroma-ingester", "INFO").new_logger()
        self.db_file_path = db_file_path
        self.collection_name = collection_name
        
    def configure_vector_store(self, initialize_db):
        db_client = chromadb.PersistentClient(path=self.db_file_path)
        if initialize_db:
            try:
                self.logger.info(f"Attempting to delete existing collection {self.collection_name}")
                db_client.delete_collection(name=self.collection_name)
                self.logger.info(f"Collection {self.collection_name} deleted")
            except Exception:
                raise ChromaCollectionDeleteError(f"Error deleting chroma collection {self.collection_name}")
        chroma_collection = db_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.logger.info(f"ChromaDB collection '{self.collection_name}' ready (current count: {chroma_collection.count()}).")
        return chroma_collection, vector_store
    
    def run_ingestion(self, folder, initialize_db):
        if not folder_exists(folder):
            raise FolderDoesNotExistError(f"Folder does not exist: {folder}")
        # Configure vector store
        chroma_collection, vector_store = self.configure_vector_store(initialize_db)
        
        case_files = list_markdown_files_in_folder(folder)
        md_parser = MarkdownCaseParser()
        # For each case, extract case number from the filename
        for case_file in case_files:
            self.logger.info(f"Processing case file: {case_file}")
            if not re.match(r"^case_\d{8}\.md$", case_file):
                raise InvalidCaseFileNameError(f"Invalid file name: {case_file}. Expected format is 'case_XXXXXXXX.md'")
            
            nodes = md_parser.process_markdown_file(f"{folder}/{case_file}")
            
            # Step 3, setup embed model then run embeddings ingestion.
            
class MarkdownCaseParser():
    def __init__(self):
        self.logger = Logger("markdown-node-parser", "INFO").new_logger()
    
    def process_markdown_file(self, file_path):
        self.logger.info(f"Parsing {file_path}")
        case_number = os.path.basename(file_path).removeprefix("case_").removesuffix(".md")
        case_filename = os.path.basename(file_path)
        
        with open(file_path, "r", encoding="utf-8") as f:
            case_file_content = f.read()
        file_metadata = {
            'case_number': case_number,
            'file_name': case_filename
        }
        doc = Document(
            text=case_file_content,
            metadata=file_metadata
        )
        parser = MarkdownNodeParser(
            include_metadata=True,
            include_prev_next_rel=True
        )
        nodes = parser.get_nodes_from_documents([doc], show_progress=True)
        
        return nodes
    
