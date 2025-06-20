import os
import logging
import re
from validators import url as valid_url
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import QueryBundle, NodeWithScore
from typing import List, Optional

class FolderDoesNotExistError(Exception):
    """Exception raised when a folder does not exist."""
    pass
class InvalidCaseFileNameError(Exception):
    """Exception raised when a case file name is invalid."""
    pass
class ChromaCollectionDeleteError(Exception):
    """Exception raised when a chroma collection couldn't be deleted."""
    pass
class InvalidAPIEndpointError(Exception):
    """Exception raised when an invalid API endpoint is provided."""
    pass
class FailedToRunChatBotWebUI(Exception):
    """Exception raised when the ChatBot webui couldn't be started."""
    pass

class Logger():
    def __init__(self, logger_name, logger_level):
        self.logger = logging.getLogger(logger_name)
        
        ch = logging.StreamHandler()
        if logger_level == "DEBUG":            
            self.logger.setLevel(logging.DEBUG)    
        elif logger_level == "ERROR":
            self.logger.setLevel(logging.ERROR)
        else:
            self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
    def new_logger(self):
        return self.logger

class SummaryKeywordMatchReranker(BaseNodePostprocessor):
    """
    Reranks nodes based on keyword matches between the query and a 'summary'
    field in the node's metadata.

    Nodes with keyword matches in their summary are boosted to the top,
    sorted by the number of keyword matches and then by their original score.
    """
    
    top_n: int = 3
    
    def __init__(self, top_n: int = 3):
        super().__init__()
        self.top_n = top_n
    
    def _get_query_keywords(self, query_str: str) -> List[str]:
        words = re.split(r'\W+', query_str.lower())
        # Basic stop words, can be expanded
        return [word for word in words if len(word) > 2 and word not in 
                ["how", "what", "is", "the", "a", "an", "do", "i", "tell", "me", "about"]]
        
    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
        if not query_bundle or not nodes: return nodes[:self.top_n] if nodes else [] # Ensure top_n is respected even if no query
        query_keywords = self._get_query_keywords(query_bundle.query_str)
        if not query_keywords: 
            return sorted(nodes, key=lambda x: x.score, reverse=True)[:self.top_n] # Assuming cosine (higher is better)
        
        boosted, others = [], []
        for n_ws in nodes:
            # Look for the summary
            summary = n_ws.node.metadata.get("summary", "").lower()
            # Give higher priority if more keywords match or if title is very similar
            match_count = sum(1 for kw in query_keywords if kw in summary)
            if match_count > 0:
                boosted.append(n_ws)
            else: 
                others.append(n_ws)
        
        # Sort boosted nodes by match count first, then by original score
        boosted.sort(key=lambda x: (sum(1 for kw in query_keywords if kw in x.node.metadata.get("summary","").lower()), x.score), reverse=True)
        others.sort(key=lambda x: x.score, reverse=True) # Sort others by original score
        
        return (boosted + others)[:self.top_n]


def folder_exists(folder_path):
    return os.path.isdir(folder_path)

def file_exists(file_path):
    return os.path.isfile(file_path)

def list_markdown_files_in_folder(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.md')]

def url_is_valid(url):
    return valid_url(url)

def extract_markdown_section_from_case_file(markdown_text, section_title):
    pattern = rf'##\s*{re.escape(section_title)}\s*\n(.*?)(\n##\s|\Z)'  # Stops at next ## or end of file
    match = re.search(pattern, markdown_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None