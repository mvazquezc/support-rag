from utils.utils import *
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.query_engine import RetrieverQueryEngine
from fastapi import FastAPI
import uvicorn
from html import unescape
import re
from llama_index.core.schema import QueryBundle
import httpx
import ssl

class RagApi():
    def __init__(self, llm_api_endpoint, embeddings_api_endpoint, model_api_key, model, context_window_length, embeddings_api_key, embeddings_model, db_file_path, db_endpoint, collection_name, api_port, skip_tls):
        self.logger = Logger("rag-api", "INFO").new_logger()
        if not url_is_valid(llm_api_endpoint):
            raise InvalidAPIEndpointError("Invalid LLM API endpoint URL")
        if not url_is_valid(embeddings_api_endpoint):
            raise InvalidAPIEndpointError("Invalid Embeddings API endpoint URL")
        if not url_is_valid(db_endpoint):
            raise InvalidAPIEndpointError("Invalid DB endpoint URL")
        self.llm_api_endpoint = llm_api_endpoint
        self.embeddings_api_endpoint = embeddings_api_endpoint
        self.model_api_key = model_api_key
        self.model = model
        self.context_window_length = context_window_length
        self.embeddings_api_key = embeddings_api_key
        self.embeddings_model = embeddings_model
        self.db_file_path = db_file_path
        self.db_endpoint = db_endpoint
        self.collection_name = collection_name
        self.api_port = api_port
        self.skip_tls = skip_tls
        
    def run(self):
        if self.skip_tls:
            self.logger.info("Skipping TLS verification.")
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            client = httpx.Client(verify=ssl_context)
            aclient = httpx.AsyncClient(verify=ssl_context)
            embeddings_model = OpenAILikeEmbedding(model_name=self.embeddings_model, api_base=self.embeddings_api_endpoint, api_key=self.embeddings_api_key, http_client=client, async_http_client=aclient)
            # is_chat_model is needed for the chatbot to work, otherwise it does completion instead of chat
            llm = OpenAILike(model=self.model, api_base=self.llm_api_endpoint, context_window=self.context_window_length, api_key=self.model_api_key, http_client=client, async_http_client=aclient, is_chat_model=True)
        else:    
            embeddings_model = OpenAILikeEmbedding(model_name=self.embeddings_model, api_base=self.embeddings_api_endpoint, api_key=self.embeddings_api_key)
            # is_chat_model is needed for the chatbot to work, otherwise it does completion instead of chat
            llm = OpenAILike(model=self.model, api_base=self.llm_api_endpoint, context_window=self.context_window_length, api_key=self.model_api_key, is_chat_model=True)
        hybrid_retriever = self.configure_hybrid_retriever(llm, embeddings_model)
        # Our final reranker, we have limited context window, we do 1. Something like 3 will be better
        reranker_top_n_results = 1
        query_engine = self.configure_query_engine(hybrid_retriever, llm, reranker_top_n_results)

        # Start the API
        app = FastAPI()

        @app.post("/answer")
        async def answer(user_query: UserQuery):
            try:
                if user_query.user_query is None:
                    return {"error": "Missing required parameter: user_query"}
                
                self.logger.info(f"Received user query: {user_query.user_query}, num_sources: {user_query.num_sources}, only_high_similarity_nodes: {user_query.only_high_similarity_nodes}")
                
                response = await self.answer_query(user_query.user_query, user_query.num_sources, query_engine, user_query.only_high_similarity_nodes)
                
                return {"response": response}
                
            except Exception as e:
                self.logger.error(f"Error processing request: {str(e)}")
                return {"error": f"Internal server error: {str(e)}"}
        
        uvicorn.run(app, host="0.0.0.0", port=self.api_port)
            
    
    async def answer_query(self, user_query, num_sources, query_engine, only_high_similarity_nodes):
        self.logger.info(f"Will use {num_sources} (current config is {query_engine._node_postprocessors[0].top_n}) for answering user query: {user_query}")
        # If the user changes the top_n results from the UI, change it for the SummaryKeywordMatchReranker
        if query_engine._node_postprocessors[0].top_n != num_sources:
            query_engine._node_postprocessors[0].top_n = num_sources
            self.logger.info(f"Changed top_n setting to {query_engine._node_postprocessors[0].top_n}")
            
        # Get the retriever and reranker components from the query engine
        retriever = query_engine.retriever
        reranker = query_engine._node_postprocessors[0] # For now we only have the SummaryKeywordMatchReranker reranker, this may need to be changed in the future

        # Perform the retrieval
        
        retrieved_nodes = await retriever.aretrieve(user_query)
        
        if only_high_similarity_nodes:
            similarity_threshold = 0.4
            filtered_nodes = []
            for node_with_score in retrieved_nodes:
                score = node_with_score.score
                self.logger.info(f"Retrieved node {node_with_score.node.id_} with score: {score:.4f} and filename {node_with_score.metadata.get('file_name')}")
                if score > similarity_threshold:
                    self.logger.info(f"Added node to the filtered nodes")
                    filtered_nodes.append(node_with_score)
        else:
            filtered_nodes = retrieved_nodes
        
        # Perform the reranking over the filtered nodes
        reranked_nodes = reranker.postprocess_nodes(
            #retrieved_nodes, query_bundle=QueryBundle(user_query) --> This does rerank on retrieved
            filtered_nodes, query_bundle=QueryBundle(user_query)
        )
        
        # From the reranked nodes get the case file name from the metadata
        case_files = []
        for node in reranked_nodes:
            self.logger.info(f"Checking node {node.id_}. Case filename is {node.metadata.get('file_name')}")
            # Append case file name if it's not in the list
            node.metadata.get('file_name') not in case_files and case_files.append(node.metadata.get('file_name'))
        
        # Send the case files required for the answer and get new nodes with the full doc instead of doc chunks
        source_nodes_for_response = self.return_full_cases_from_vectordb_as_node(case_files)
        
        if len(source_nodes_for_response) == 0:
            return "No cases found that can be used to answer your query"
        # The following block can be uncommented if you want to know what text was sent
        # to the LLM context window (primarily for debugging purposes)
        
        #context_list = [node.get_content() for node in source_nodes_for_response]
        #context_str = "\n\n---------------------\n\n".join(context_list)
    
        #self.logger.info("--- START CONTEXT BEING SENT TO LLM ---")
        #self.logger.info(context_str)
        #self.logger.info("--- END CONTEXT BEING SENT TO LLM ---")

        # Perform the final synthesis step to get the answer
        
        response = await query_engine.asynthesize(user_query, nodes=source_nodes_for_response)
        
        # Extract text and clean HTML entities
        cleaned_response = unescape(response.response)
        # Remove <think> section using regex
        cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL)
        cleaned_response = cleaned_response.replace('```', '\n```')
        cleaned_response = cleaned_response + "\n\n**Cases used for the answer:** "
        
        for source_node in response.source_nodes:
            clean_title = source_node.node.metadata.get('case_number', 'N/A')
            cleaned_response = cleaned_response + "\n  - " + clean_title
        return str(cleaned_response)
    
    def configure_query_engine(self, hybrid_retriever, llm, reranker_top_n_results):
        qa_template_str = (
#            "Question: {query_str}\n"
            "You are an assistant with access to previously opened support cases.\n"
            "These support cases are markdown files with the following structure:\n"
            "# <case_title>\n"
            "## Summary\n"
            "<case_summary_text>\n"
            "## Description\n"
            "<case_description_text>\n"
            "## Comments\n"
            "### Comment 1\n"
            "<comment_1_text>\n"
            "### Comment N\n"
            "<comment_N_text>\n"
            
            "Using the provided context, you must analyze the different comments and identify the ones that led to a solution:\n"
            "- Once you have the comments, analyze all the comments and think step by step.\n"
            "- Extract the solution from the comments and send that information to the user.\n"
            "- The solution MUST include all the required steps to fix the issue, make sure to include the steps in the correct order.\n"
            "- If the solution is not found, state that you don't know or that the information is not available in the provided context.\n"

            "If you cannot find a solution within the comments, state that you don't know or that the information is not available in the provided context.\n\n"
            
            "Context:\n{context_str}\n\n"

            "Answer:"
        )
        qa_prompt = PromptTemplate(qa_template_str)
        
        # Reranking by reranker model should happen here as a node_postprocessor, for now we have summary keyword...
        #reranker = SentenceTransformerRerank(
        #   model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        #    top_n=reranker_top_n_results
        #)
        
        summary_keyword_reranker = SummaryKeywordMatchReranker(top_n=reranker_top_n_results)
        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever, # Use the hybrid retriever
            llm=llm,
            node_postprocessors=[summary_keyword_reranker], # Reranker assigned to hybrid results
            text_qa_template=qa_prompt,
        )
        
        return query_engine
    
    def return_full_cases_from_vectordb_as_node(self, case_filenames):
        if self.db_endpoint is not None:
            _, host, port = split_url_endpoint(self.db_endpoint)
            db_client = chromadb.HttpClient(host=host, port=port)
        else:
            db_client = chromadb.PersistentClient(path=self.db_file_path)
        chroma_collection = db_client.get_collection(name=self.collection_name)
        source_nodes_for_response = []
        for case_file in case_filenames:
            self.logger.info(f"Gathering nodes for case filename {case_file} from VectorDB")
            all_chunks_data = chroma_collection.get(
                where={"file_name": case_file},
                include=["documents", "metadatas"] # Fetch text and metadata
            )
            # Combine metadata and documents for easy sorting
            combined_chunks = zip(all_chunks_data["metadatas"], all_chunks_data["documents"])
            # Sort the chunks by chunk_index
            sorted_chunks = sorted(combined_chunks, key=lambda item: item[0].get('chunk_index', 0))
            # Reconstruct the document by joining the sorted chunk texts
            document_texts = [text for metadata, text in sorted_chunks]
            full_document_text = "\n\n".join(document_texts)
            # Create a source node for citation
            node_metadata = {
                'file_name': case_file,
                'case_number': os.path.basename(case_file).removeprefix("case_").removesuffix(".md")
            }
            full_doc_node = TextNode(text=full_document_text, metadata=node_metadata)
            source_nodes_for_response.append(NodeWithScore(node=full_doc_node, score=1))
        
        return source_nodes_for_response
        
    def configure_hybrid_retriever(self, llm, embeddings_model):
        # Configure BM25Retriever, we need to get nodes from VectorDB first
        self.logger.info("Initializing BM25Retriever.")
        if self.db_endpoint is not None:
            _, host, port = split_url_endpoint(self.db_endpoint)
            db_client = chromadb.HttpClient(host=host, port=port)
        else:
            db_client = chromadb.PersistentClient(path=self.db_file_path)
        chroma_collection = db_client.get_collection(name=self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        chroma_results = chroma_collection.get()
        nodes = []
        for i in range(len(chroma_results["ids"])):
            node = TextNode(
                id_=chroma_results["ids"][i],
                text=chroma_results["documents"][i],
                metadata=chroma_results["metadatas"][i]
                # Note: Embeddings are not needed for BM25, so we don't need to load them.
            )
            nodes.append(node)
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
        self.logger.info(f"BM25Retriever initialized. Retrieved {len(nodes)} from ChromaDB.")
        # Initialize Semantic Retriever
        self.logger.info("Initializing SemanticRetriever.")
        vector_store_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embeddings_model)
        semantic_retriever = vector_store_index.as_retriever(similarity_top_k=5)
        self.logger.info("SemanticRetriever initialized.")
        
        fusion_retriever = QueryFusionRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            similarity_top_k=5, # How many results to return in the final list
            # Reciprocal Rerank works best with generated queries (Query vs Passage)
            num_queries=1,
            #mode="reciprocal_rerank",
            mode="dist_based_score",
            # Simple rerank works well with just the original query
            #num_queries=1,
            #mode="simple",
            use_async=True,
            verbose=True,
            llm=llm,
            # You can also configure weights for each retriever, e.g., [0.6, 0.4]
            retriever_weights=[0.5, 0.5]
        )
        
        return fusion_retriever
    
