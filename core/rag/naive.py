import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
)
from qdrant_client.local.qdrant_local import QdrantLocal
from llama_index.vector_stores import QdrantVectorStore
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding

class NaiveRAG():
    def __init__(self, data_path:str,  db_path:str, collection_name:str='demo_collection'):
        self.db_path = db_path
        self.PERSIST_DIR = data_path
        self.collection_name = collection_name
        self.__openai_key = "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"
        
        self.llm = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.0, api_key=self.__openai_key)
        self.embed_model = OpenAIEmbedding(api_key=self.__openai_key)
        self.service_context = ServiceContext.from_defaults(llm=self.llm, embed_model=self.embed_model)
        
        self.vector_store = QdrantVectorStore(
            client=QdrantLocal(location=self.db_path),
            collection_name=self.collection_name,)
        
    def __get_documents(self):
        if os.path.exists(self.PERSIST_DIR):
            # load the documents and create the index
            self.documents = SimpleDirectoryReader(self.PERSIST_DIR).load_data()
        else:
            print(f"Directory {self.PERSIST_DIR} does not exist")
        return self.documents
    
    def __store_documents(self):
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def __vector_store_index(self):
        self.index = VectorStoreIndex.from_documents(
            documents=self.documents,
            service_context=self.service_context,
            storage_context=self.storage_context,
        )
    
    def __create_engine(self):
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3,
            vector_store_query_mode="default",
            # filters=MetadataFilters(
            #     filters=[
            #         ExactMatchFilter(key="name", value="paul graham"),
            #     ]
            # ),
            alpha=None,
            doc_ids=None,
        )
    
    def run_naive(self):
        self.__get_documents()
        self.__store_documents()
        self.__vector_store_index()
        self.__create_engine()
    
    def query_vector_storage(self):
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            service_context=self.service_context,
        )
        self.__create_engine()
    
    def query(self, query:str):
        return self.query_engine.query(query)
    
    def get_prompt(self):
        return self.query_engine._get_prompts()
    
