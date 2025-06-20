# Support Case RAG Tool

This is a PoC that aims to provide a RAG tool that can answer user queries by using previous support case data to back its answers.

## Support Case Files Structure

The support case files ingested by the tool are expected to be formatted in Markdown and to have the following structure:

~~~code
# <case_number> - <case_title>
## Summary
<case_summary>
## Description
<case_description>
## Comments
### Comment 1
<comment_1>
### Comment N
<comment_n>
~~~

The files are expected to have the following naming:

* case_<case_number>.md

## How to run the tool

1. Install requirements:

    ~~~sh
    python -m venv .venv
    pip install -r requirements.txt
    ~~~

2. Run data ingestion:

    > **NOTE**: By default the ingestion will only ingest data not existing in the Vector DB, if you want to initialize the db from scratch you can add `-i` parameter.

    ~~~sh
    python src/cli.py ingest -d /folder/with/case/files -m <embeddings_model> -e <ollama_endpoint>

    #Example:
    python src/cli.py ingest -d ./case_files -m nomic-embed-text:latest -e http://127.0.0.1:11434
    ~~~

3. Run ChatBot:

    > **NOTE**: By default the chatbot WebUI will listen on port 8080, you can change the port by using the `-p` parameter.

    ~~~sh
    python src/cli.py chatbot -m <llm_model> -e <ollama_endpoint>/ -em <embeddings_model>
    #Example
    python src/cli.py chatbot -m gemma3:4b -e http://127.0.0.1:11434 -em nomic-embed-text:latest
    ~~~
