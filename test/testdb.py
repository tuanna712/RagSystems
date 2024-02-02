from core.connector.qdrantdb import MyQdrant

My_Qdrant = MyQdrant(local_location='database')
My_Qdrant.create_collection('demo_collection')
