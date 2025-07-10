from utils.utils import *
from ingestion.ingest import ChromaIngester
from chatbot.chatbot import GradioChatBot
from agent.agent import GradioAgent
from ragapi.ragapi import RagApi
import argparse
import os


def run_chatbot(args, logger):
    """Function to handle the 'chatbot' action."""
    logger.info(f"Starting chatbot on port {args.port}.")
    try:
        chatbot = GradioChatBot(chatbot_port=args.port, rag_api_endpoint=args.rag_api_endpoint, skip_tls=args.insecure_skip_tls)
        chatbot.run()
    except Exception as e:
        logger.error(f"{e}")
        return 1

def run_agent(args, logger):
    """Function to handle the 'agent' action."""
    logger.info(f"Starting agent on port {args.port}.")
    try:
        agent = GradioAgent(agent_port=args.port, rag_api_endpoint=args.rag_api_endpoint, llm_api_endpoint=args.llm_api_endpoint, model_api_key=args.llm_api_key, model=args.model, context_window_length=args.context_window_length, skip_tls=args.insecure_skip_tls)
        agent.run()
    except Exception as e:
        logger.error(f"{e}")
        return 1

def run_rag_api(args, logger):
    """Function to handle the 'rag-api' action."""
    logger.info(f"Starting rag api on port {args.port}.")
    try:
        rag_api = RagApi(llm_api_endpoint=args.llm_api_endpoint, embeddings_api_endpoint=args.embeddings_api_endpoint, model_api_key=args.llm_api_key, model=args.model, context_window_length=args.context_window_length, embeddings_api_key=args.embedding_api_key, embeddings_model=args.embeddings_model, db_file_path="./chromadb", db_endpoint=args.db_endpoint, collection_name="support_cases", api_port=args.port, skip_tls=args.insecure_skip_tls)
        rag_api.run()
    except Exception as e:
        logger.error(f"{e}")
        return 1

def run_local_ingest(args, logger):
    """Function to handle the 'ingest' action."""
    logger.info(f"Starting ingestion from folder {args.source_dir}.")
    try:
        ingester = ChromaIngester(db_file_path="./chromadb", db_endpoint=args.db_endpoint, collection_name="support_cases", embeddings_api_endpoint=args.embeddings_api_endpoint, embeddings_api_key=args.embedding_api_key, embeddings_model=args.embeddings_model, skip_tls=args.insecure_skip_tls)
        ingester.run_local_ingestion(args.source_dir, args.initialize_db)
    except FolderDoesNotExistError as e:
        logger.error(f"{e}")
        return 1
    except InvalidCaseFileNameError as e:
        logger.error(f"{e}")
        return 1
    except ChromaCollectionDeleteError as e:    
        logger.error(f"{e}")
        return 1
    except InvalidAPIEndpointError as e:
        logger.error(f"{e}")
        return 1

def run_s3_ingest(args, logger):
    """Function to handle the 's3-ingest' action."""
    logger.info(f"Starting ingestion from s3 bucket {args.s3_bucket} and path {args.s3_path}.")
    try:
        ingester = ChromaIngester(db_file_path="./chromadb", db_endpoint=args.db_endpoint, collection_name="support_cases", embeddings_api_endpoint=args.embeddings_api_endpoint, embeddings_api_key=args.embedding_api_key, embeddings_model=args.embeddings_model, skip_tls=args.insecure_skip_tls)
        ingester.run_s3_ingestion(args.s3_bucket, args.s3_path, args.s3_endpoint, args.initialize_db, args.s3_access_key, args.s3_secret_key, args.interval, args.insecure_skip_tls, args.s3_download_folder)
    except Exception as e:
        logger.error(f"{e}")
        return 1

def main():
    
    # Create logger for the cli
    logger = Logger("support-rag-cli", "INFO").new_logger()

    # Create the top-level parser
    parser = argparse.ArgumentParser(
        prog="support-rag-cli",
        description='RAG app for support cases.'
    )
    # Create the subparser
    subparsers = parser.add_subparsers(dest="action", required=True, help="Available actions")
    

    # Add rag-api parser
    parser_rag_api = subparsers.add_parser("rag-api", help="Initiate rag api.")
    # Add listen port argument
    parser_rag_api.add_argument(
        "-p",
        "--port",
        type=int,
        required=False,
        default=8080,
        help="The port where the WebUI will be exposed."
    )
    parser_rag_api.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="The LLM model to be used."
    )
    parser_rag_api.add_argument(
        "-w",
        "--context-window-length",
        required=False,
        default=10000,
        type=str,
        help="The model's context window length."
    )
    parser_rag_api.add_argument(
        "-em",
        "--embeddings-model",
        required=True,
        type=str,
        help="The embeddings model to be used."
    )
    parser_rag_api.add_argument(
        "-llm-api",
        "--llm-api-endpoint",
        required=True,
        type=str,
        help="The api endpoint to access the llm model."
    )
    parser_rag_api.add_argument(
        "-em-api",
        "--embeddings-api-endpoint",
        required=True,
        type=str,
        help="The api endpoint to access the embeddings model."
    )
    parser_rag_api.add_argument(
        "--db-endpoint",
        required=False,
        type=str,
        help="The endpoint to access the vector database."
    )
    parser_rag_api.add_argument(
        "-lk",
        "--llm-api-key",
        required=True,
        type=str,
        help="The api key to access the llm model. Default value reads OPENAI_API_KEY env var."
    )
    parser_rag_api.add_argument(
        "-ek",
        "--embedding-api-key",
        required=True,
        type=str,
        help="The api key to access the embedding model. Default value reads OPENAI_EMBEDDING_API_KEY env var."
    )   
    parser_rag_api.add_argument(
        "--insecure-skip-tls",
        required=False,
        action='store_true',
        default=False,
        help="If set, TLS connections to api endpoints skip cert verification."
    )
    # Set function that handles the chatbot action
    parser_rag_api.set_defaults(func=run_rag_api)

    # Add chatbot parser
    parser_chatbot = subparsers.add_parser("chatbot", help="Initiate chat webui.")
    # Add listen port argument
    parser_chatbot.add_argument(
        "-p",
        "--port",
        type=int,
        required=False,
        default=8181,
        help="The port where the WebUI will be exposed."
    )
    parser_chatbot.add_argument(
        "--rag-api-endpoint",
        required=False,
        default="http://127.0.0.1:8080/answer",
        type=str,
        help="The endpoint to access the rag api."
    )
    parser_chatbot.add_argument(
        "--insecure-skip-tls",
        required=False,
        action='store_true',
        default=False,
        help="If set, TLS connections to api endpoints skip cert verification."
    )
    # Set function that handles the chatbot action
    parser_chatbot.set_defaults(func=run_chatbot)
    # Add chatbot parser
    parser_agent = subparsers.add_parser("agent", help="Initiate agent webui.")
    # Add listen port argument
    parser_agent.add_argument(
        "-p",
        "--port",
        type=int,
        required=False,
        default=8282,
        help="The port where the WebUI will be exposed."
    )
    parser_agent.add_argument(
        "--rag-api-endpoint",
        required=False,
        default="http://127.0.0.1:8080/answer",
        type=str,
        help="The endpoint to access the rag api."
    )
    parser_agent.add_argument(
        "--insecure-skip-tls",
        required=False,
        action='store_true',
        default=False,
        help="If set, TLS connections to api endpoints skip cert verification."
    )
    parser_agent.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="The LLM model to be used."
    )
    parser_agent.add_argument(
        "-w",
        "--context-window-length",
        required=False,
        default=10000,
        type=str,
        help="The model's context window length."
    )
    parser_agent.add_argument(
        "-llm-api",
        "--llm-api-endpoint",
        required=True,
        type=str,
        help="The api endpoint to access the llm model."
    )
    parser_agent.add_argument(
        "-lk",
        "--llm-api-key",
        required=True,
        type=str,
        help="The api key to access the llm model. Default value reads OPENAI_API_KEY env var."
    )
    # Set function that handles the chatbot action
    parser_agent.set_defaults(func=run_agent)
    # Add local-ingest parser
    parser_local_ingest = subparsers.add_parser("local-ingest", help="Ingest data from a local source.")
    # Add source dir argument
    parser_local_ingest.add_argument(
        "-d",
        "--source-dir",
        required=True,
        type=str,
        help="The directory containing the data to ingest."
    )
    parser_local_ingest.add_argument(
        "-m",
        "--embeddings-model",
        required=True,
        type=str,
        help="The embeddings model to be used."
    )
    parser_local_ingest.add_argument(
        "--db-endpoint",
        required=False,
        type=str,
        help="The endpoint to access the vector database."
    )
    parser_local_ingest.add_argument(
        "-em-api",
        "--embeddings-api-endpoint",
        required=True,
        type=str,
        help="The api endpoint to access the embedding model."
    )
    parser_local_ingest.add_argument(
        "-ek",
        "--embedding-api-key",
        required=True,
        type=str,
        help="The api key to access the embedding model. Default value reads OPENAI_EMBEDDING_API_KEY env var."
    )   
    parser_local_ingest.add_argument(
        "-i",
        "--initialize-db",
        required=False,
        action='store_true',
        default=False,
        help="If set, the vector database will be cleaned before running the ingestion."
    )
    parser_local_ingest.add_argument(
        "--insecure-skip-tls",
        required=False,
        action='store_true',
        default=False,
        help="If set, TLS connections to api endpoint skip cert verification."
    )
    # Set function that handles the ingest action
    parser_local_ingest.set_defaults(func=run_local_ingest)
    # Add local-ingest parser
    parser_s3_ingest = subparsers.add_parser("s3-ingest", help="Ingest data from a s3 source.")
    # Add source dir argument

    parser_s3_ingest.add_argument(
        "-m",
        "--embeddings-model",
        required=True,
        type=str,
        help="The embeddings model to be used."
    )
    parser_s3_ingest.add_argument(
        "-em-api",
        "--embeddings-api-endpoint",
        required=True,
        type=str,
        help="The api endpoint to access the embedding model."
    )
    parser_s3_ingest.add_argument(
        "-ek",
        "--embedding-api-key",
        required=True,
        type=str,
        help="The api key to access the embedding model. Default value reads OPENAI_EMBEDDING_API_KEY env var."
    )   
    parser_s3_ingest.add_argument(
        "-i",
        "--initialize-db",
        required=False,
        action='store_true',
        default=False,
        help="If set, the vector database will be cleaned before running the ingestion."
    )
    parser_s3_ingest.add_argument(
        "--db-endpoint",
        required=False,
        type=str,
        help="The endpoint to access the vector database."
    )
    parser_s3_ingest.add_argument(
        "--insecure-skip-tls",
        required=False,
        action='store_true',
        default=False,
        help="If set, TLS connections to api endpoint skip cert verification."
    )
    parser_s3_ingest.add_argument(
        "-b",
        "--s3-bucket",
        required=True,
        type=str,
        help="The S3 bucket to ingest data from."
    )
    parser_s3_ingest.add_argument(
        "-p",
        "--s3-path",
        required=True,
        type=str,
        help="The S3 path to ingest data from."
    )
    parser_s3_ingest.add_argument(
        "-e",
        "--s3-endpoint",
        required=False,
        type=str,
        help="The S3 endpoint to ingest data from."
    )
    parser_s3_ingest.add_argument(
        "-ak",
        "--s3-access-key",
        required=False,
        default=os.environ.get('S3_ACCESS_KEY'),
        type=str,
        help="The S3 acccess key."
    )
    parser_s3_ingest.add_argument(
        "-sk",
        "--s3-secret-key",
        required=False,
        default=os.environ.get('S3_SECRET_KEY'),
        type=str,
        help="The S3 secret key."
    )
    parser_s3_ingest.add_argument(
        "--interval",
        required=False,
        default=5,
        type=int,
        help="The interval in minutes to ingest data from the S3 bucket."
    )
    parser_s3_ingest.add_argument(
        "--s3-download-folder",
        required=False,
        default="./s3-download",
        type=str,
        help="The folder to download the S3 files to."
    )
    # Set function that handles the ingest action
    parser_s3_ingest.set_defaults(func=run_s3_ingest)
    args = parser.parse_args()
    args.func(args,logger)
    
if __name__ == "__main__":
    main()



