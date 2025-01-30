'''
This class is a slightly modified version of the original Langchain BM25ElasticSearch Retriever. Mainly we add the
possibility to create a retrieval object without loading files or creating an index in ElasticSearch DocumentStore
'''

from __future__ import annotations
import os 
import uuid
from tqdm import tqdm
from typing import Any, Iterable, List, Optional
from elasticsearch import Elasticsearch

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import AzureOpenAIEmbeddings
import pandas as pd



class CustomElasticSearch(BaseRetriever):
    """`Elasticsearch` retriever that uses `BM25`.

    Minor modifications from ElasticSearchBM25 from langchain.elasticsearch
    - Uses text.page_content instead of only text.
    - Tries and catches exception instead of returning an error the author tries to create again an index which already exists.
    - create method takes argument recreate: If True the index will be deleted and recreated.
    """

    client: Any
    index_name: str
    embedding_model_name: Optional[str]
    embedding_model: Optional[AzureOpenAIEmbeddings]
    hybrid: Any
    """Name of the index to use in Elasticsearch."""

    def __init__(self, index_name: str, client: Elasticsearch = None, embedding_model_name=None,
                 hybrid: bool = False) -> None:
        super().__init__(index_name=index_name)
        self.client = client
        self.embedding_model_name = embedding_model_name
        if self.embedding_model_name == "text-embedding-ada-002":
            self.embedding_model = AzureOpenAIEmbeddings(
                                    api_key=os.environ["OPENAI_API_KEY"],
                                    azure_endpoint=os.environ["OPENAI_AZURE_ENDPOINT"],
                                    azure_deployment=os.environ["AZURE_EMBEDDING_DEPLOYMENT"],
                                    model=self.embedding_model_name)
        else:
            self.embedding_model = None
        if embedding_model_name is None and hybrid is True:
            raise ValueError('Cannot apply hybrid mode without embedding model')
        self.hybrid = hybrid

    @classmethod
    def create(
            cls, elasticsearch_url: str, index_name: str, recreate: bool, k1: float = 2.0, b: float = 0.75, embedding_model_name: str = None,
            hybrid: bool = False
    ) -> CustomElasticSearch:
        """
        Create a ElasticSearchBM25Retriever from a list of texts.

        Args:
            elasticsearch_url: URL of the Elasticsearch instance to connect to.
            index_name: Name of the index to use in Elasticsearch.
            k1: BM25 parameter k1.
            b: BM25 parameter b.
            embedding_model_name: embedding model to use
        Returns:
        """

        # Create an Elasticsearch client instance
        es = Elasticsearch(elasticsearch_url)

        # Define the index settings and mappings
        settings = {
            "analysis": {"analyzer": {"default": {"type": "standard"}}},
            "similarity": {
                "custom_bm25": {
                    "type": "BM25",
                    "k1": k1,
                    "b": b,
                }
            },
        }
        if embedding_model_name is None:
            mappings = {
                "properties": {
                    "content": {
                        "type": "text",
                        "similarity": "custom_bm25",  # Use the custom BM25 similarity
                        }
                    }
                }
        elif hybrid is False:
            mappings = {
                "properties": {
                    "content": {
                        "type": "text",
                        "similarity": "custom_bm25",  # Use the custom BM25 similarity
                        }
                    }
                }
        else:
            mappings = {
                "properties": {
                    "content": {
                        "type": "text",
                        "similarity": "custom_bm25",  # Use the custom BM25 similarity
                    },
                    "vector": {
                        "type": "dense_vector",
                        "dims": 1536,
                        "index": True,
                        "similarity": "dot_product",
                    }
                }
            }

        # Create the index with the specified settings and mappings
        if recreate:
            try:
                es.indices.delete(index=index_name)  ### NEW
            except Exception as e:
                print(f"No index to delete.")

        try:
            es.indices.create(index=index_name, mappings=mappings, settings=settings)
        except Exception as e:
            print(e)
        return cls(client=es, index_name=index_name, embedding_model_name=embedding_model_name, hybrid=hybrid)

    @classmethod
    def delete_index(
            elasticsearch_url: str,
            index_name: str):
        es = Elasticsearch(elasticsearch_url)
        es.indices.delete(index_name)

    def add_texts(
            self,
            texts: Iterable[str],
            refresh_indices: bool = True,

    ) -> List[str]:
        """Run more texts through the embeddings and add to the retriever.

        Args:
            texts: Iterable of strings to add to the retriever.
            refresh_indices: bool to refresh ElasticSearch indices

        Returns:
            List of ids from adding the texts into the retriever.
        """
        try:
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ValueError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        requests = []
        ids = []
        for i, text in tqdm(enumerate(texts)):
            _id = str(uuid.uuid4())
            if self.embedding_model:
                request = {
                    "_op_type": "index",
                    "_index": self.index_name,
                    'tag': text.metadata['tag'],
                    'file': text.metadata['file'],
                    'content': text.page_content,
                    'vector': self.embedding_model.embed_documents([text.page_content])[0],
                    '_id': _id,
                }
            else:
                request = {
                    '_op_type': "index",
                    '_index': self.index_name,
                    'tag': text.metadata['tag'],
                    'file': text.metadata['file'],
                    'content': text.page_content,
                    '_id': _id,
                    }
            ids.append(_id)
            requests.append(request)

        bulk(self.client, requests)
        if refresh_indices:
            self.client.indices.refresh(index=self.index_name)
        return ids

    def retrieve(self, query: str, file: str, run_manager: CallbackManagerForRetrieverRun = None, n_retrieved: int = 10):
        return self._get_relevant_documents(query, file, run_manager, n_retrieved)

    def _get_relevant_documents(
            self, query: str, file: str, run_manager: CallbackManagerForRetrieverRun, n_retrieved: int
    ) -> List[Document]:
        if self.hybrid is False:
            if self.embedding_model:
                # print(f"Search with embeddings with:\n {machine_code}\n {query}\n {n_retrieved}")
                query_emb = self.embedding_model.embed_query(query)
                query_dict = {
                    "field": "vector",
                    "query_vector": query_emb,
                    "k": n_retrieved,
                    "num_candidates": 20,
                    "filter": [
                        {"match": {"file": file}},
                    ]
                }
                res = self.client.search(index=self.index_name, knn=query_dict)
                docs = []
                for r in res["hits"]["hits"]:
                    docs.append(Document(page_content=r["_source"]["content"]))
                return docs
            else:
                query_dict = {"query":{
                "bool": {
                    "must": [{"match": {"file": file}}],
                    "should":[{"match": {"content": query}}],
                    },
                }, "size": n_retrieved}

                res = self.client.search(index=self.index_name, body=query_dict)

                docs = []
                for r in res["hits"]["hits"]:
                    docs.append(Document(page_content=r["_source"]["content"]))
                return docs
        else:
            df_docs = pd.DataFrame(columns=['id', 'document', 'score_bm25', 'score_emb', 'score_rrf'])

            #Retrieve with BM25
            query_dict = {"query": {
                "bool": {
                    "must": [{"match": {"file": file}}],
                    "should": [{"match": {"content": query}}],
                },
            }, "size": 5}
            res_BM25 = self.client.search(index=self.index_name, body=query_dict)
            for r in res_BM25["hits"]["hits"]:

                new_row = {'id': (r["_id"]),
                           'document': Document(page_content=r["_source"]["content"]),
                           'score_bm25': (r['_score']),
                           'score_emb': 0.,
                           'score_rrf': 0.
                }
                df_docs = df_docs._append(new_row, ignore_index=True)
            df_docs = df_docs.set_index('id')

            #Retrieve with embeddings
            query_emb = self.embedding_model.embed_query(query)
            query_dict = {
                "field": "vector",
                "query_vector": query_emb,
                "k": 50,
                "num_candidates": 500,
                "filter": [
                    {"match": {"tag": machine_code}},
                ]
            }
            res_emb = self.client.search(index=self.index_name, knn=query_dict, size=50)
            for r in res_emb["hits"]["hits"]:
                if r['_id'] in df_docs.index:
                    df_docs.at[r['_id'], 'score_emb'] = r['_score']
                else:
                    df_docs.loc[r["_id"]] = [Document(page_content=r["_source"]["content"]), 0, (r['_score']), 0]
            k = 10
            df_docs['score_rrf'] = 1/(df_docs['score_emb']+k)*1/(df_docs['score_bm25']+k)
            df_docs = df_docs.sort_values('score_rrf', ascending=True)
            docs = []
            i = 0
            for index, row in df_docs.iterrows():
                if i == n_retrieved:
                    break
                docs.append(row['document'])
                i += 1
            return docs

