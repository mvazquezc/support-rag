import os
import logging
import re
import boto3
from validators import url as valid_url
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import QueryBundle, NodeWithScore
from typing import List, Optional
from urllib.parse import urlparse
from pydantic import BaseModel

class UserQuery(BaseModel):
    user_query: Optional[str] = None
    num_sources: int = 1
    only_high_similarity_nodes: bool = False

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
class S3Error(Exception):
    """Exception raised when an S3 error occurs."""
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

def split_url_endpoint(url):
    parsed_url = urlparse(url)
    return parsed_url.scheme, parsed_url.hostname, parsed_url.port

def extract_markdown_section_from_case_file(markdown_text, section_title):
    pattern = rf'##\s*{re.escape(section_title)}\s*\n(.*?)(\n##\s|\Z)'  # Stops at next ## or end of file
    match = re.search(pattern, markdown_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

class S3:

    def __init__(self, s3_access_key, s3_secret_key, s3_endpoint, skip_tls):
        logging = Logger("s3_utils", "DEBUG")
        self.logger = logging.new_logger()
        if s3_endpoint is not None:
            verify_ssl=True
            if skip_tls:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                verify_ssl=False
            self.client = boto3.client('s3', endpoint_url=s3_endpoint, verify=verify_ssl, aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key)
        else:
            self.client = boto3.client('s3', aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key)
        
    def upload(self, s3_bucket, file_object, object_name, folder=None):
        if folder is not None:
            object_name = folder + '/' + object_name
        self.client.upload_file(file_object, s3_bucket, object_name)

    def move_file(self, s3_bucket, s3_file_path, new_s3_file_path):
        # remove folder if exists from destination name
        self.client.copy_object(CopySource=f'{s3_bucket}/{s3_file_path}', Bucket=s3_bucket, Key=new_s3_file_path)
        self.client.delete_object(Bucket=s3_bucket, Key=s3_file_path)

    def download(self, s3_bucket, s3_file_path, output_file_path):
        self.client.download_file(s3_bucket, s3_file_path, output_file_path)

    def delete_object(self, s3_bucket, s3_file_path):
        self.client.delete_object(Bucket=s3_bucket, Key=s3_file_path)

    def list_bucket_content(self, s3_bucket, s3_path=None, s3_filter=None):
        keys = []
        if s3_path is not None:
            if not s3_path.endswith('/'):
                s3_path += '/'
            kwargs = {'Bucket': s3_bucket, 'Prefix': s3_path}
        else:
            kwargs = {'Bucket': s3_bucket}
        while True:
            try:
                resp = self.client.list_objects_v2(**kwargs)
                if resp['ResponseMetadata']['HTTPStatusCode'] == 200 and resp['KeyCount'] != 0:           
                    try:
                        for obj in resp['Contents']:
                            keys.append(obj['Key'])
                    except KeyError:
                        raise S3Error("Error while listing bucket content")
                    try:
                        kwargs['ContinuationToken'] = resp['NextContinuationToken']
                    except KeyError:
                        break
                elif resp['ResponseMetadata']['HTTPStatusCode'] != 200:
                    raise S3Error("Error while listing bucket content. Make sure S3 Bucket and Path are correct")
                else:
                    self.logger.warning(f"Bucket {s3_bucket} is empty")
                    break
            except self.client.exceptions.NoSuchBucket:
                raise S3Error(f"Bucket {s3_bucket} does not exist")
        if s3_filter is not None and len(keys) > 0:
            for entry in keys[:]:
                if not (entry.endswith(s3_filter)):
                    keys.remove(entry)
        return keys

    def download_markdown_files_from_bucket(self, s3_bucket, output_folder, folder=None):
        # Create a list with only .md files present on s3 bucket
        markdown_files = self.list_bucket_content(s3_bucket, folder, ".md")
        if len(markdown_files) > 0:
            for s3_markdown_file_path in markdown_files:
                markdown_file_path, markdown_file_filename = os.path.split(s3_markdown_file_path)
                output_file_path = os.path.join(output_folder, markdown_file_filename)
                if os.path.isfile(output_file_path):
                   self.logger.debug(f"Markdown file {output_file_path} already downloaded")
                else:
                    # Create output folder if it does not exist
                    if not (os.path.isdir(output_folder)):
                        self.logger.info(f"Folder {output_folder} does not exist, creating it for the first time...")
                        os.makedirs(output_folder)
                    self.download(s3_bucket, s3_markdown_file_path, output_file_path)
                    self.logger.debug(f"Markdown file {s3_markdown_file_path} downloaded to {output_file_path}")
        return markdown_files