from utils.utils import *
import os
import re
import chromadb
import hashlib
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.ingestion import IngestionPipeline


class ChromaIngester():
    def __init__(self, db_file_path, collection_name, api_endpoint, embeddings_model):
        self.logger = Logger("chroma-ingester", "INFO").new_logger()
        self.db_file_path = db_file_path
        self.collection_name = collection_name
        if not url_is_valid(api_endpoint):
            raise InvalidAPIEndpointError("Invalid API endpoint URL")
        self.api_endpoint = api_endpoint
        self.embeddings_model = embeddings_model
        
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
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_construction": 300,
                    "ef_search": 300,
                }
            }
            #metadata={"hnsw:space": "cosine"}
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.logger.info(f"ChromaDB collection '{self.collection_name}' ready (current count: {chroma_collection.count()}).")
        return chroma_collection, vector_store
    
    def run_embeddings_ingestion(self, embeddings_model, vector_store, parsed_nodes, chroma_collection):
        pipeline = IngestionPipeline(
            transformations=[embeddings_model], 
            vector_store=vector_store,
        )
        self.logger.info("Running embeddings ingestion.")
        pipeline.run(nodes=parsed_nodes, show_progress=True)
        self.logger.info(f"Ingestion completed. items in ChromaDB: {chroma_collection.count()}")
    
    def run_ingestion(self, folder, initialize_db):
        if not folder_exists(folder):
            raise FolderDoesNotExistError(f"Folder does not exist: {folder}")
        # Configure vector store
        self.logger.info("Vector store configured.")
        chroma_collection, vector_store = self.configure_vector_store(initialize_db)
        # Configure embeddings model
        embeddings_model = OllamaEmbedding(model_name=self.embeddings_model, base_url=self.api_endpoint)
        self.logger.info("Embeddings model configured.")
        case_files = list_markdown_files_in_folder(folder)
        md_parser = MarkdownCaseParser()
        # For each case, extract case number from the filename
        case_files_to_process = len(case_files)
        processed_files = 1
        for case_file in case_files:
            self.logger.info(f"Processing case file: {case_file}. [{processed_files}/{case_files_to_process}]")
            if not re.match(r"^case_\d{8}\.md$", case_file):
                raise InvalidCaseFileNameError(f"Invalid file name: {case_file}. Expected format is 'case_XXXXXXXX.md'.")
            
            try:
                nodes = md_parser.process_markdown_file(f"{folder}/{case_file}")
            except Exception:
                # TODO: change exception to custom one
                raise ChromaCollectionDeleteError(f"Error parsing markdown for nodes from {case_file}")
            try:
                self.run_embeddings_ingestion(embeddings_model, vector_store, nodes, chroma_collection)
                # Step 3, setup embed model then run embeddings ingestion.
                processed_files += 1
            except Exception as e:
                # TODO: change exception to custom one
                raise ChromaCollectionDeleteError(f"Error running embeddings ingestion: {e}")
                
            
class MarkdownCaseParser():
    def __init__(self):
        self.logger = Logger("markdown-node-parser", "INFO").new_logger()
    
    def process_markdown_file(self, file_path):
        self.logger.info(f"Parsing {file_path}.")
        case_number = os.path.basename(file_path).removeprefix("case_").removesuffix(".md")
        case_filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            case_file_content = f.read()
        # Get summary from markdown and add it as metadata for future title-based reranking
        case_summary = extract_markdown_section_from_case_file(case_file_content, "Summary")
        file_metadata = {
            'case_number': case_number,
            'file_name': case_filename,
            'case_summary': case_summary,
        }
        doc = Document(
            text = case_file_content,
            metadata = file_metadata
        )
        parser = MarkdownNodeParser(
            include_metadata=True,
            include_prev_next_rel=True
        )
        nodes = parser.get_nodes_from_documents([doc], show_progress=True)
        # Avoid duplicated nodes in the vector db
        for i, node in enumerate(nodes):
            id_string_for_hash = f"{case_number}-{i}-{node.get_content()}"
            node_id = hashlib.sha256(id_string_for_hash.encode()).hexdigest()
            node.id_ = node_id
            node.metadata["chunk_index"] = i

        return nodes
    
