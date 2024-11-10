from pymilvus import MilvusClient,connections, FieldSchema, CollectionSchema, DataType, Collection
import json

class MilvusDbConnection:
    def __init__(self,db_name,collection_name):

        self.db=db_name
        self.collection_name = collection_name

        self.client = MilvusClient(
            uri="http://localhost:19530",
            db_name=db_name
        )
        # create collection if not
        if not self.client.has_collection(collection_name=self.collection_name):
            self.create_schema(self.collection_name)

    
    def create_schema(self,collection_name,description="RAG QA Collection"):

        # # Define field schemas
        primary_key_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        title_field = FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512)
        meta_field = FieldSchema(name="meta", dtype=DataType.VARCHAR, max_length=512)
        text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048)
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)

        # # Create collection schema
        collection_schema = CollectionSchema(
            fields=[primary_key_field, title_field, meta_field, text_field, embedding_field],
            description=description
        )

        self.client.create_collection(
            collection_name=collection_name,
            schema=collection_schema
        )

        index_params = MilvusClient.prepare_index_params()

        index_params.add_index(
            field_name="embedding",
            metric_type="COSINE",
            index_type="FLAT",
            index_name="embedding_index",
            params={ "nlist": 1024 }
        )

        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )

        return 

    def insert_data(self,data):
        self.client.insert(collection_name=self.collection_name, data=data)
    
    def search(self,data):
        res = self.client.search(
            collection_name=self.collection_name,
            data=[data],
            limit=5, # Max. number of search results to return
            search_params={"metric_type": "COSINE", "params": {'radius':-0.5}}, # Search parameters
            output_fields=["title","meta","text"]
        )

        return res[0]
            
    def close_connection(self):
        self.client.close()

