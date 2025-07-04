from utils.utils import *
import gradio as gr
from gradio import ChatMessage
import aiohttp
import ssl

class GradioChatBot():
    def __init__(self, chatbot_port, rag_api_endpoint, skip_tls):
        self.logger = Logger("gradio-chatbot", "INFO").new_logger()
        self.chatbot_port = chatbot_port
        if not url_is_valid(rag_api_endpoint):
            raise InvalidAPIEndpointError("Invalid RAG API endpoint URL")
        self.rag_api_endpoint = rag_api_endpoint
        self.skip_tls = skip_tls
        
    def run(self):
        
        reranker_top_n_results = 1
        
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
                response = await self.answer_query(last_user_message, num_sources, only_high_similarity_nodes)
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
            
    
    async def answer_query(self, user_query, num_sources, only_high_similarity_nodes):
        # Create the query parameters
        query_params = {
            "user_query": user_query,
            "num_sources": num_sources,
            "only_high_similarity_nodes": only_high_similarity_nodes
        }
        
        # Make API call to the RAG API endpoint
        try:
            async with aiohttp.ClientSession() as session:
                # Use the RAG API endpoint
                api_url = self.rag_api_endpoint
                
                # Configure SSL context if skip_tls is True
                if self.skip_tls:
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    connector = aiohttp.TCPConnector(ssl=ssl_context)
                    session = aiohttp.ClientSession(connector=connector)
                
                async with session.post(api_url, json=query_params) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "No response received from API")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"API call failed with status {response.status}: {error_text}")
                        return f"Error: API call failed with status {response.status}"
                        
        except Exception as e:
            self.logger.error(f"Failed to call RAG API: {e}")
            return f"Error: Failed to call RAG API - {str(e)}"
       
    
   
