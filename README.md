# Description
This repository is a base template for genrative AI (LLM) with a retrieval system (RAG) using elastic search Vector DB
This base project is using the open source model LLAMA-2 downloaded from hugging face compatible with a CPU infrastructure.
You can find the different models used and their configurations under: app/settings.py

- Put your dataset for RAG context under the data/ folder
- app scripts are under app/ :
    - document_manager.py: Retrieve the documents to be embedded.
    - embedding_manager.py: Embedding and indexing 
    - search_engine.py: Similarity search with vector DB.
    - settings.py: models and app settings.
    - text_splitter.py: Chunking documents.
    - model_loader: Loading LLM.
    - main.py: Main app script.

# Requirements
- Python: >= 3.9
- Venv or Conda for your python virtual environment.
- A hugging face account and an authentication token to download the open source models. (Check the hugging face section below).
- Langchain framework.
- Elastic search

# Hugging Face and model download
- Get your hugginface account and CLI: https://huggingface.co/docs/huggingface_hub/en/guides/cli
- Generate your huggingface token for login and installations of open source models: https://huggingface.co/settings/tokens 
- Get access to llama from meta: https://huggingface.co/meta-llama/Llama-2-7b-hf (wait for review first)
- Models are downloaded from here: https://huggingface.co/TheBloke (using the CLI command below)
- Important: Hugging Face account email MUST match the email you provide to Meta when you request approval.


| Name | Quant method | Bits | Size | Max RAM required | Use case
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| llama-2-7b.Q4_K_M.gguf | Q4_K_M | 4  | 4.08 GB  | 6.58 GB	medium | balanced quality - recommended  |

# Create a virtual environment
you can use conda or venv:

- Conda
```
conda create -n <your-env-name> python=<your-python-version>
conda activate <your-env-name>
```

- Venv
```
python3 -m venv <your-env-name>
source <your-env-name>/bin/activate
```

# Installation
- Install the requirements
```
pip3 install -r requirements.txt
```

- Install the model

```
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf --local-dir models/ --local-dir-use-symlinks False
```
Models should be installed under the folder models/ in the root of this project.

I am using "meta-llama/Llama-2-7b-hf" instead of "TheBloke/Llama-2-7B-GGUF" because the tokenizer model is not included in the GGUF converted LLM. (the LLM and the tokenizer need to be compatible). In this case, both are of type "llama".

# Execution
- Run:
```  python demo.py ```


# More details
- FAISS does not support cloud deployment but there are other options you can choose from:
https://medium.com/the-ai-forum/which-vector-database-should-you-use-choosing-the-best-one-for-your-needs-5108ec7ba133

- You can use Open AI GPT with Langchain or any other model.


# Extra details: elastic search embeddings and vector search
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.12.2
docker pull docker.elastic.co/kibana/kibana:8.12.2
https://github.com/context-labs/mactop

```
docker network create elastic

docker run -d --name elasticsearch --net elastic -p 9200:9200 -p 9300:9300  -e "discovery.type=single-node" -e "ELASTIC_PASSWORD=root" docker.elastic.co/elasticsearch/elasticsearch:8.12.2

docker run -d --name kibana --net elastic -p 5601:5601 docker.elastic.co/kibana/kibana:8.12.2
```
