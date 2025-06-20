from utils.utils import *
from ingestion.ingest import ChromaIngester
from chatbot.chatbot import GradioChatBot
import argparse


def run_chatbot(args, logger):
    """Function to handle the 'chatbot' action."""
    logger.info(f"Starting chatbot on port {args.port}.")
    try:
        chatbot = GradioChatBot(api_endpoint=args.api_endpoint, model=args.model, context_window_length=args.context_window_length, embeddings_model=args.embeddings_model, db_file_path="./chromadb", collection_name="support_cases", chatbot_port=args.port)
        chatbot.run()
    except Exception as e:
        logger.error(f"{e}")
        return 1

def run_ingest(args, logger):
    """Function to handle the 'ingest' action."""
    logger.info(f"Starting ingestion from folder {args.source_dir}.")
    try:
        ingester = ChromaIngester(db_file_path="./chromadb", collection_name="support_cases", api_endpoint=args.api_endpoint, embeddings_model=args.embeddings_model)
        ingester.run_ingestion(args.source_dir, args.initialize_db)
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
    
    # Add chatbot parser
    parser_chatbot = subparsers.add_parser("chatbot", help="Initiate chat webui.")
    # Add listen port argument
    parser_chatbot.add_argument(
        "-p",
        "--port",
        type=int,
        required=False,
        default=8080,
        help="The port where the WebUI will be exposed."
    )
    parser_chatbot.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="The LLM model to be used."
    )
    parser_chatbot.add_argument(
        "-w",
        "--context-window-length",
        required=False,
        default=10000,
        type=str,
        help="The model's context window length."
    )
    parser_chatbot.add_argument(
        "-em",
        "--embeddings-model",
        required=True,
        type=str,
        help="The embeddings model to be used."
    )
    parser_chatbot.add_argument(
        "-e",
        "--api-endpoint",
        required=True,
        type=str,
        help="The api endpoint to access the embedding model."
    )   
    # Set function that handles the chatbot action
    parser_chatbot.set_defaults(func=run_chatbot)
    # Add ingest parser
    parser_ingest = subparsers.add_parser("ingest", help="Ingest data from a source.")
    # Add source dir argument
    parser_ingest.add_argument(
        "-d",
        "--source-dir",
        required=True,
        type=str,
        help="The directory containing the data to ingest."
    )
    parser_ingest.add_argument(
        "-m",
        "--embeddings-model",
        required=True,
        type=str,
        help="The embeddings model to be used."
    )
    parser_ingest.add_argument(
        "-e",
        "--api-endpoint",
        required=True,
        type=str,
        help="The api endpoint to access the embedding model."
    )
    parser_ingest.add_argument(
        "-i",
        "--initialize-db",
        required=False,
        action='store_true',
        #type=bool,
        default=False,
        help="If set, the vector database will be cleaned before running the ingestion."
    )
    # Set function that handles the ingest action
    parser_ingest.set_defaults(func=run_ingest)
    args = parser.parse_args()
    args.func(args,logger)
    
if __name__ == "__main__":
    main()



