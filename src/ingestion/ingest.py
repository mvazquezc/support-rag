from utils.utils import *
import os
import re
import chromadb
import hashlib
import httpx
import ssl
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
import time
import shutil

class ChromaIngester():
    def __init__(self, db_file_path, db_endpoint, collection_name, embeddings_api_endpoint, embeddings_api_key, embeddings_model, skip_tls):
        self.logger = Logger("chroma-ingester", "INFO").new_logger()
        self.db_file_path = db_file_path
        self.db_endpoint = db_endpoint
        self.collection_name = collection_name
        if not url_is_valid(embeddings_api_endpoint):
            raise InvalidAPIEndpointError("Invalid API endpoint URL")
        if not url_is_valid(db_endpoint):
            raise InvalidAPIEndpointError("Invalid DB endpoint URL")
        self.embeddings_api_endpoint = embeddings_api_endpoint
        self.embeddings_api_key = embeddings_api_key
        self.embeddings_model = embeddings_model
        self.skip_tls = skip_tls

    def configure_vector_store(self, initialize_db):
        if self.db_endpoint is not None:
            _, host, port = split_url_endpoint(self.db_endpoint)
            db_client = chromadb.HttpClient(host=host, port=port)
        else:
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
    
    def run_s3_ingestion(self, s3_bucket, s3_path, s3_endpoint, initialize_db, s3_access_key, s3_secret_key, interval, skip_tls, s3_download_folder):
        if not s3_bucket:
            raise S3Error("S3 bucket is required")
        if not s3_path:
            raise S3Error("S3 path is required")
        if not s3_endpoint:
            raise S3Error("S3 endpoint is required")
        if not s3_access_key:
            raise S3Error("S3 access key is required")
        if not s3_secret_key:
            raise S3Error("S3 secret key is required")
        # Create S3 client
        s3 = S3(s3_access_key, s3_secret_key, s3_endpoint, skip_tls)
        # Since this is a daemon, we need to initialize the db only once
        initialize_db_once = initialize_db
        # Create ingested folder if it does not exist
        if not os.path.exists(s3_download_folder):
            os.makedirs(s3_download_folder)
        while True:
            # Download files
            files_downloaded = s3.download_markdown_files_from_bucket(s3_bucket, s3_download_folder, s3_path)
            
            if len(files_downloaded) > 0:
                self.logger.info(f"Downloaded {len(files_downloaded)} files from S3 bucket.")
                # Ingest files
                self.run_local_ingestion(s3_download_folder, initialize_db_once)
                # Remove ingested files from path
                shutil.rmtree(s3_download_folder)
                # Move ingested files to ingested folder
                for file in files_downloaded:
                    s3.move_file(s3_bucket, file, f"ingested/{os.path.basename(file)}")
                self.logger.info(f"Ingestion completed. Sleeping for {interval} minutes before checking S3 bucket for new files.")
            else:
                self.logger.info(f"No new files found. Sleeping for {interval} minutes before checking S3 bucket for new files.")
            time.sleep(interval * 60)
            initialize_db_once = False
        
    def run_local_ingestion(self, folder, initialize_db):
        if not folder_exists(folder):
            raise FolderDoesNotExistError(f"Folder does not exist: {folder}")
        # Configure vector store
        self.logger.info("Vector store configured.")
        chroma_collection, vector_store = self.configure_vector_store(initialize_db)
        # Configure embeddings model
        if self.skip_tls:
            self.logger.info("Skipping TLS verification.")
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            client = httpx.Client(verify=ssl_context)
            embeddings_model = OpenAILikeEmbedding(model_name=self.embeddings_model, api_base=self.embeddings_api_endpoint, api_key=self.embeddings_api_key, http_client=client)
        else:    
            embeddings_model = OpenAILikeEmbedding(model_name=self.embeddings_model, api_base=self.embeddings_api_endpoint, api_key=self.embeddings_api_key)
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
    
