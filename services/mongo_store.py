# services/mongo_store.py
import logging
from typing import List, Optional, Dict, Any
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ServerSelectionTimeoutError
from config.settings import settings
from loguru import logger

_client = None
_coll = None

def get_collection():
    global _client, _coll
    if _coll is not None:
        return _coll
    if not settings.mongo.uri:
        raise RuntimeError("Set MONGODB_URI in environment variables")
    _client = MongoClient(settings.mongo.uri, serverSelectionTimeoutMS=5000)
    try:
        _client.admin.command("ping")
    except ServerSelectionTimeoutError as e:
        raise RuntimeError(f"Cannot connect to MongoDB: {e}")
    db = _client[settings.mongo.database]
    _coll = db[settings.mongo.collection]

    # helpful indexes
    try:
        _coll.create_index([("service", ASCENDING), ("timestamp", DESCENDING)])
        _coll.create_index([("id", ASCENDING), ("timestamp", DESCENDING)])
    except Exception as e:
        logger.warning(f"Index creation failed: {e}")

    return _coll

def insert_event(record: Dict[str, Any]) -> None:
    coll = get_collection()
    coll.insert_one(record)
    logger.info("Event inserted successfully")

def list_latest_events_for_service(service: str, limit: int) -> List[Dict[str, Any]]:
    coll = get_collection()
    pipeline = [
        {"$match": {"service": service}},
        {"$sort": {"timestamp": -1}},
        {"$group": {"_id": "$id", "doc": {"$first": "$$ROOT"}}},
        {"$replaceRoot": {"newRoot": "$doc"}},
        {"$sort": {"timestamp": -1}},
        {"$limit": int(limit)},
    ]
    records = list(coll.aggregate(pipeline))
    # Convert ObjectId to string for JSON serialization
    for record in records:
        if '_id' in record:
            record['_id'] = str(record['_id'])
    return records

def get_latest_event_for_process(process_id: str) -> Optional[Dict[str, Any]]:
    coll = get_collection()
    doc = coll.find_one({"id": process_id}, sort=[("timestamp", DESCENDING)])
    if doc and '_id' in doc:
        doc['_id'] = str(doc['_id'])
    return doc