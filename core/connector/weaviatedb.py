import os, weaviate
import weaviate.classes as wvc
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.config import (
    Property, 
    DataType, 
    Tokenization,
    Configure,
    VectorDistances,
    Reconfigure,
)

class ClientWeaviate:
    def __init__(self, URL:str="", API_KEY:str="", mode:str="cloud"):
        # Connect to a WCS instance
        if mode == "cloud":
            self.client = weaviate.connect_to_wcs(
                cluster_url=URL,
                auth_credentials=weaviate.auth.AuthApiKey(API_KEY),
                skip_init_checks=True,
                )
        if mode == "local":
            self.client = weaviate.connect_to_local()
            
    def get_schema(self): # List all collections
        return self.client.collections.list_all(simple=False)
    
    def connect(self):
        self.client.connect()

    def is_live(self):
        return self.client.is_live()
    
    def is_connected(self):
        return self.client.is_connected()
    
    def disconnect(self):
        self.client.close()

    def delete_collection(self, _collection_name):
        if (self.client.collections.exists(_collection_name)):
            self.client.collections.delete(_collection_name)  # Replace with your collection name
            print("Collection {} deleted!".format(_collection_name))
        else:
            print("Collection {} does not exist!".format(_collection_name))

    def delete_all_collections(self):
        self.client.collections.delete_all()
        print("All collections deleted!")

    def auto_create_collection(self, new_collection_name):
        self.client.collections.create(new_collection_name)
        print(f"Collection created! {new_collection_name}")

    def create_collection(self, new_collection_name, property_list):
        self.client.collections.create(
            new_collection_name,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(), # Using OpenAI
            vector_index_config=Configure.VectorIndex.hnsw(
                dynamic_ef_factor=10,
                distance_metric=VectorDistances.COSINE,
            ), # Set vector index type: "hnsw" or "flat" index types. Compression ("pq" for "hnsw" indexes and "bq" for "flat" indexes)
            inverted_index_config=Reconfigure.inverted_index(
                bm25_k1=1.1,
                bm25_b=0.8
            ),
            # generative_config=Configure.Generative.openai(
            #     model="gpt-3.5-turbo-1106",
            #     max_tokens=2000,
            #     temperature=0.0,
            # ),
            # replication_config=Configure.replication(
            #     factor=3,
            # ), # Configure replication per collection
            properties=property_list,
            # Property(name="title", 
            #     description="Title of the article",
            #     data_type=DataType.TEXT,
            #     vectorize_property_name=True,  # Use "title" as part of the value to vectorize
            #     tokenization=Tokenization.WORD  # Use "word" tokenization
            #     indexFilterable=True,
            #     indexSearchable=True,
            # ),
        )
        
class WVCollection(ClientWeaviate):
    def __init__(self, URL:str="", API_KEY:str="", collection_name:str="demo_collection", mode:str="cloud"):
        super().__init__(URL, API_KEY, mode)
        self.collection_name = collection_name
        if self.client.collections.exists(self.collection_name):
            self.collection = self.client.collections.get(self.collection_name)
            self.total_count = self.collection.aggregate.over_all(total_count=True).total_count
        else:
            print(f"Collection {self.collection_name} does not exist!")

    def get_config(self):
        return self.collection.config.get()
    
    def insert_obj(self, obj_uuid, properties, vector):
        self.collection.data.insert(
            uuid=obj_uuid,
            properties=properties,
            vector=vector,
            # references=wvc.data.Reference("f81bfe5e-16ba-4615-a516-46c2ae2e5a80"),  # If you want to add a reference (if configured in the collection definition)
        )

    def get_object_by_id(self, obj_uuid):
        return self.collection.query.fetch_object_by_id(uuid=obj_uuid)
        
    def update_obj(self, obj_uuid, properties, vector):
        self.collection.data.update(
            uuid=obj_uuid,
            properties=properties,
            vector=vector,
        )

    def replace_obj(self, obj_uuid, properties, vector):
        self.collection.data.replace(
            uuid=obj_uuid,
            properties=properties,
            vector=vector,
        )

    def delete_obj(self, obj_uuid):
        self.collection.data.delete_by_id(uuid=obj_uuid)

    def search(self, 
               user_query, 
               query_vector, 
               limit:str=3, 
               alpha:float=0.25, 
               auto_limit:int=1, 
               filters:wvc.query.Filter=None, 
               ):
        return self.collection.query.hybrid(
            query=user_query,
            # query_properties=["question^2", "answer"], #Specify properties to keyword search - '^2' is a boost/weight factor
            vector=query_vector,
            limit=limit,
            alpha=alpha, #Balance keyword and vector search, 1 is a pure vector search, 0 is a pure keyword search
            fusion_type=wvc.query.HybridFusion.RELATIVE_SCORE, #Change the ranking method - "RANKED" or "RELATIVE_SCORE
            # auto_limit=auto_limit, #Limit results to groups with similar distances from the query,
            filters=filters,
            # filters=wvc.query.Filter.by_property("round").contains_any(["Double Jeopardy!"]), #https://weaviate.io/developers/weaviate/api/graphql/filters#filter-structure
            return_metadata=wvc.query.MetadataQuery(score=True, explain_score=True),
        )