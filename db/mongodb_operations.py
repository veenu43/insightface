from pymongo import MongoClient


def get_database(dbname):
    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    CONNECTION_STRING = "mongodb+srv://nusmtechse21project:project123$@cluster0.vjyr2lj.mongodb.net/test"

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    client = MongoClient(CONNECTION_STRING)

    # Create the database for our example (we will use the same database throughout the tutorial
    return client[dbname]


def save_to_db(dbName, collectionName, collections):
    dbname = get_database(dbName)
    print(dbname)
    collection_name = dbname[collectionName]
    print(collection_name.insert_many(collections))


item_1 = {
    "id": 1660193719,
    "timestamp": "11-08-2022 12:55:19",
    "status": "Status",
    "model_backend": "mtcnn",
    "recognition_model": "DeepFace",
    "model_metrics": "cosine",
    "accuracy_metric": {
        "match_rate": 90,
        "metric_type": "Average",
        "metric": 0.6,
        "threshold": 0.4
    }
}

item_2 = {
    "id": 2,
    "timestamp": "2022-05-08 18:44:41",
    "status": "SUCCESS",
    "backendModel": "mtcnn",
    "recognitionModel": "DeepFace",
    "metrics": {
        "name": "cosine",
        "version": 1
    },
    "dbDataset": {
        "version": 1,
        "count": 10000
    },
    "input": {
        "version": 1,
        "count": 100
    },
    "accuracy": {
        "matchRate": 90,
        "metricType": "Average",
        "metric": 0.6,
        "threshhold": 0.4
    }
}
# save_to_db('faceRecognition', 'model_run', [item_1])
