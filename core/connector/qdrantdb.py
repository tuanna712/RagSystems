from qdrant_client import QdrantClient
from qdrant_client.local.qdrant_local import QdrantLocal
from qdrant_client.http import models

class MyQdrant():
    def __init__(self, qdrant_url:str=None, qdrant_api_key:str=None, local_location:str=None):
        if local_location==None:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            self.client = QdrantLocal(location=local_location)
    def create_collection(self, collection_name:str, embedding_size:int=1536):
        try:
            self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=embedding_size, distance=models.Distance.COSINE),
                )
            print(f"Collection {collection_name} created")
        except:
            pass

    def recreate_collection(self, collection_name:str, embedding_size:int=1536):
        self.client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=embedding_size, distance=models.Distance.COSINE),
                )
        print(f"Collection {collection_name} recreated")

    def delete_collection(self, collection_name:str):
        self.client.delete_collection(collection_name=collection_name)

    def count_data(self, collection_name:str):
        return self.client.count(collection_name=collection_name)

    def upsert_data(self, points:list, collection_name:str):
        self.client.upsert(collection_name=collection_name, points=points)

    def scroll_data(self, collection_name:str):
        return self.client.scroll(collection_name)
    
    def search_data(self, collection_name:str, query_vector:list, top_k:int=10):
        return self.client.search(collection_name, query_vector, limit=top_k)