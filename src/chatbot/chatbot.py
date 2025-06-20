from utils.utils import *
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine
import gradio as gr
from gradio import ChatMessage
from html import unescape
import re
from llama_index.core.schema import QueryBundle

class GradioChatBot():
    def __init__(self, api_endpoint, model, context_window_length, embeddings_model, db_file_path, collection_name, chatbot_port):
        self.logger = Logger("gradio-chatbot", "INFO").new_logger()
        if not url_is_valid(api_endpoint):
            raise InvalidAPIEndpointError("Invalid API endpoint URL")
        self.api_endpoint = api_endpoint
        self.model = model
        self.context_window_length = context_window_length
        self.embeddings_model = embeddings_model
        self.db_file_path = db_file_path
        self.collection_name = collection_name
        self.chatbot_port = chatbot_port
        
    def run(self):
        
        llm = Ollama(model=self.model, base_url=self.api_endpoint, context_window=self.context_window_length, request_timeout=300.0)
        hybrid_retriever = self.configure_hybrid_retriever(llm)
        # Our final reranker, we have limited context window, we do 1. Something like 3 will be better
        reranker_top_n_results = 1
        query_engine = self.configure_query_engine(hybrid_retriever, llm, reranker_top_n_results)
        with gr.Blocks(theme="soft", title="Support Cases Chatbot 💬", fill_height=True) as webui:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Settings")
                    num_sources = gr.Slider(1, 5, value=1, step=1,
                                  label="Number of sources for reranking",
                                  info="Choose how many sources should be returned by the reranker")
                    only_high_similarity_nodes = gr.Checkbox(False,label="Use only cases with high similarity",info="Disregard cases that are not highly similar with the submitted query")

                with gr.Column(scale=4):
                    gr.Markdown("# Support Cases Chatbot 💬")
                    gr.Markdown("Ask me questions about previous support cases")

                    chatbot = gr.Chatbot(
                        value=[],
                        height=None,
                        type="messages",
                        min_height=400,
                        max_height=1000,
                        elem_id="chatbot",
                        container=True,
                        render_markdown=True,
                    )
                    textbox = gr.Textbox(
                        placeholder="Type your question here and press Enter",
                        container=False,
                        scale=7
                    )


            # Step 1: Add user message to chat box
            def add_user_and_placeholder(message, chat_history):
                chat_history = chat_history or []
                chat_history.append(ChatMessage(role="user", content=message))
                return "", chat_history

            # Step 2: Generate real response and replace placeholder
            async def generate_response(chat_history, num_sources, only_high_similarity_nodes):
                if not chat_history:
                    return chat_history
                last_user_message = chat_history[-1]['content']
                response = await self.answer_query(last_user_message, num_sources, query_engine, only_high_similarity_nodes)
                chat_history.append(ChatMessage(role="assistant", content=response))
                return chat_history

            # Once user submits the query run the required functions
            textbox.submit(
                add_user_and_placeholder,
                inputs=[textbox, chatbot],
                outputs=[textbox, chatbot]
            ).then(
                fn=generate_response,
                inputs=[chatbot, num_sources, only_high_similarity_nodes],
                outputs=[chatbot]
            )
        try:
            webui.launch(server_port=self.chatbot_port)
        except:
             raise FailedToRunChatBotWebUI("ChatBot WebUI failed to start")
            
    
    
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
            "Question: {query_str}\n"
            "Context:\n{context_str}\n\n"
            
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
            
            "Your goal is:\n"
            "Using the provided context, analyze the different comments and identify the ones that led to a solution\n"
            "Once you have the comments, extract the solution from them and send that information to the user\n"
            "The solution MUST include all the required steps to fix the issue\n"
            "If you cannot find a solution within the comments, state that you don't know or that the information is not available in the provided context.\n\n"
            
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
        
    def configure_hybrid_retriever(self, llm):
        # Configure BM25Retriever, we need to get nodes from VectorDB first
        self.logger.info("Initializing BM25Retriever.")
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
        embeddings_model = OllamaEmbedding(model_name=self.embeddings_model, base_url=self.api_endpoint)        
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
    
