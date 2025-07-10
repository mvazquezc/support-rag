from utils.utils import *
import gradio as gr
from gradio import ChatMessage
import aiohttp
import ssl
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import AgentRunner, ReActAgentWorker
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage as LlamaIndexChatMessage
import httpx


class GradioAgent():
    def __init__(self, agent_port, rag_api_endpoint, llm_api_endpoint, model_api_key, model, context_window_length, skip_tls):
        self.logger = Logger("gradio-agent", "INFO").new_logger()
        self.agent_port = agent_port
        if not url_is_valid(rag_api_endpoint):
            raise InvalidAPIEndpointError("Invalid RAG API endpoint URL")
        if not url_is_valid(llm_api_endpoint):
            raise InvalidAPIEndpointError("Invalid LLM API endpoint URL")
        self.rag_api_endpoint = rag_api_endpoint
        self.llm_api_endpoint = llm_api_endpoint
        self.model_api_key = model_api_key
        self.model = model
        self.context_window_length = context_window_length
        self.skip_tls = skip_tls
        self.ticket_state = {}
        self.num_sources = 1
        self.only_high_similarity_nodes = False

        if self.skip_tls:
            self.logger.info("Skipping TLS verification.")
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            client = httpx.Client(verify=ssl_context)
            aclient = httpx.AsyncClient(verify=ssl_context)
            # is_chat_model is needed for the chatbot to work, otherwise it does completion instead of chat
            llm = OpenAILike(model=self.model, api_base=self.llm_api_endpoint, context_window=self.context_window_length, api_key=self.model_api_key, http_client=client, async_http_client=aclient, is_chat_model=True)
        else:    
            # is_chat_model is needed for the chatbot to work, otherwise it does completion instead of chat
            llm = OpenAILike(model=self.model, api_base=self.llm_api_endpoint, context_window=self.context_window_length, api_key=self.model_api_key, is_chat_model=True)


        support_ticket_tool = FunctionTool.from_defaults(fn=self.search_support_tickets)
        update_state_tool = FunctionTool.from_defaults(fn=self.update_ticket_state)

        agent_worker = ReActAgentWorker(
            tools=[support_ticket_tool, update_state_tool],
            llm=llm,
            verbose=True, # Set to True to see the agent's thought process
            max_iterations=100 
        )
        self.agent = AgentRunner(
            agent_worker=agent_worker,
            verbose=True, # Set to True to see the agent's thought process
        )


        self.chat_history: List[LlamaIndexChatMessage] = []
        self.system_prompt = """You are a helpful support ticket assistant. Your goal is to collect specific pieces of information from the user to create a complete support case query.

On the first turn, you must use the `update_ticket_state` tool to store the original user message in the `user_description` field. Store the original user message as is, without any modifications.

After the first turn, on each turn, follow these steps:
1. **Observe**: Understand what information the user is providing about their support issue.
2. **Reason**: Identify what information is still missing. The required fields are: `product_version`, `user_description`, `error_message`.
3. **Act (Update State)**: When the user provides information, use the `update_ticket_state` tool to save it. For example:
   - If user says "I'm using version 2.1", use update_ticket_state with field="product_version" and value="2.1"
   - If user says "I get error 404", use update_ticket_state with field="error_message" and value="error 404"
4. **Act (Search)**: Use the `search_support_tickets` tool to check if we have all required information. The tool will automatically ask for missing information or perform the search when complete.

When updating product_version, the version MUST be a valid version string, for example 1.0, v1.0, etc. You MUST NOT ASUME ANY VERSION, ASK THE USER FOR IT IF IT IS NOT PROVIDED.

**CRITICAL: Use tools to answer the user's question. ALWAYS USE TOOLS TO ANSWER THE USER'S QUESTION.**

**CRITICAL: When the `search_support_tickets` tool returns search results (not asking for more information), you MUST respond with ONLY those results. Do NOT use any other tools after receiving search results. The search results are the final answer to the user's question.**

Remember to be conversational and helpful throughout the interaction."""
        
        # Initialize chat history with the system prompt
        self.chat_history = [
            LlamaIndexChatMessage(role="system", content=self.system_prompt)
        ]

    async def respond(self, message: str, history: List[List[str]]) -> tuple[List[List[str]], str]:
        """
        Handle user messages and generate responses using the agent.
        
        Args:
            message (str): The user's message
            history (List[List[str]]): The chat history in Gradio format
            
        Returns:
            tuple: Updated history and empty string (for new message)
        """
        try:
            # Get response from agent
            response = await self.agent.achat(message, chat_history=self.chat_history)
            
            # Update our internal chat history
            self.chat_history.append(LlamaIndexChatMessage(role="user", content=message))
            self.chat_history.append(LlamaIndexChatMessage(role="assistant", content=str(response)))
            
            # Return the updated history (Gradio will handle the display)
            return history + [[message, str(response)]], ""
            
        except Exception as e:
            error_message = f"An error occurred: {e}"
            print(error_message)
            return history + [[message, error_message]], ""
    
    async def get_agent_response(self, message: str) -> str:
        """
        Get response from agent without updating the chat history.
        
        Args:
            message (str): The user's message
            
        Returns:
            str: The agent's response
        """
        try:
            # Get response from agent
            response = await self.agent.achat(message, chat_history=self.chat_history)
            
            # Update our internal chat history
            self.chat_history.append(LlamaIndexChatMessage(role="user", content=message))
            self.chat_history.append(LlamaIndexChatMessage(role="assistant", content=str(response)))
            
            return str(response)
            
        except Exception as e:
            error_message = f"An error occurred: {e}"
            print(error_message)
            return error_message
    
    def clear_chat(self):
        """Clear the chat history and reset the state."""
        self.ticket_state.clear()
        self.chat_history = [
            LlamaIndexChatMessage(role="system", content=self.system_prompt)
        ]
        return [], "", "No information gathered yet"

    def update_ticket_state(self, field: str, value: str) -> str:
        """
        Update the ticket query state with information provided by the user.

        Args:
            field (str): The field to update (product_version, error_message, or environment)
            value (str): The value provided by the user

        Returns:
            str: Confirmation message
        """
        if field not in ["product_version", "error_message", "user_description"]:
            return f"Invalid field: {field}. Valid fields are: product_version, error_message, user_description"

        self.ticket_state[field] = value
        print(f"Updated {field}: {value}")
        print("Current state:", self.ticket_state)
        return f"Updated {field} to: {value}"

    async def search_support_tickets(self, query: str) -> str:
        """
        This function searches for support tickets based on the provided query.
        If not all required information is available in the TICKET_QUERY_STATE,
        it will ask the user for the missing information one piece at a time.

        Args:
            query (str): The user's query. This is required by the agent but not
                         directly used to fill the state. The LLM uses the conversation
                         history to populate the state.

        Returns:
            str: A message to the user, either asking for more information or
                 confirming that the search is being performed.
        """
        print("TICKET INFO: ")
        print(self.ticket_state)
        # Check for missing information in our state
        if not self.ticket_state.get("product_version"):
            return "I can help with that. What is the product version you are using?"
        if not self.ticket_state.get("error_message"):
            return "Thanks. What is the exact error message you are seeing?"

        # If we have all the information, we can now "call" our API
        print(f"✅ All information gathered: {self.ticket_state}")
        print("🚀 Querying the support ticket API...")

        query =  f"Original User query: {self.ticket_state.get('user_description')}\n Error Message: {self.ticket_state.get('error_message')}\n Product Version: {self.ticket_state.get('product_version')}"
        response = await self.answer_query(query, self.num_sources, self.only_high_similarity_nodes)
        # Simulate an API call with the gathered information
        # In a real application, you would replace this with your actual API call.

        # Clear the state for the next interaction
        self.ticket_state.clear()

        return response

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
                # Add extra params below, need to update the other function
                self.num_sources = num_sources
                self.only_high_similarity_nodes = only_high_similarity_nodes
                response = await self.get_agent_response(last_user_message)
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
            webui.launch(server_port=self.agent_port, server_name="0.0.0.0")
        except:
             raise FailedToRunChatBotWebUI("Agent WebUI failed to start")
            
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
       
    
   
