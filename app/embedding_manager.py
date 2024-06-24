from langchain_community.embeddings import HuggingFaceEmbeddings
from elasticsearch import Elasticsearch
from langchain_community.vectorstores import ElasticsearchStore

from langchain_community.vectorstores import FAISS
from . import settings

class EmbeddingManager:
    def __init__(self, model_path, device, normalize_embeddings):
        self.es = Elasticsearch(
            [settings.ES_URL],
            verify_certs=False,
            ssl_show_warn=False,
            basic_auth=(settings.USERNAME, settings.PASSWORD)
        )
        self.index = settings.INDEX
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': normalize_embeddings}
            )

    # faiss vector store
    def get_vectorstore(self, docs):
        # vectors = self.embeddings.embed_documents([doc.page_content for doc in docs])
        # for i, vector in enumerate(vectors):
        #     print(f"Vector {i}: {vector}")
        
        return FAISS.from_documents(docs, self.embeddings)
    

    def get_elastic_db(self):
       return ElasticsearchStore(
            es_connection=self.es,
            embedding=self.embeddings,
            index_name=self.index,
            )


    ## elastic search
    def create_index(self):
        # Create an index with a vector field
        mapping = {
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": 384  # The dimension of your embeddings
                    },
                    "text": {
                        "type": "text"
                    }
                }
            },
        }

        # Create the index (delete if exists)
        if self.es.indices.exists(index=self.index):
            self.es.indices.delete(index=self.index)
        self.es.indices.create(index=self.index, body=mapping)

    def index_documents(self, docs, metadata):
        # Get embeddings for the documents
        vectors = self.embeddings.embed_documents([doc.page_content for doc in docs])
        print(docs)
        for i, (doc, vector) in enumerate(zip(docs, vectors)):
            doc_body = {
                'vector': vector,
                'text': doc.page_content,
                'metadata': doc.metadata
            }
            self.es.index(index=self.index, body=doc_body)


    def display_vectors(self):
        mapping = self.es.indices.get_mapping(index=self.index)
        print(mapping)
        query = {
            "query": {
                "match_all": {}
            },
            "size": 10
        }

        response = self.es.search(index=self.index, body=query)
        print("response")
        print(response['hits']['hits'])
        if response['hits']['hits']:
            for hit in response['hits']['hits']:
                print(f"ID: {hit['_id']}, Vector: {hit['_source'].get('vector')}, Text: {hit['_source'].get('text')}")
        else:
            print("No documents found.")