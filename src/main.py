from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

def main():
    try:
        # Connect to local MongoDB (default host/port, no auth)
        client = MongoClient("mongodb://NebuZard:7YAMTHHD@localhost:27017/", serverSelectionTimeoutMS=3000)

        # Optional: ping the server to check connection
        client.admin.command("ping")
        print("✅ Connected to MongoDB!")

    except ConnectionFailure as e:
        print("❌ Could not connect to MongoDB:", e)
        return

    # 2) Choose a database & collection (MongoDB creates them on first write)
    db = client["my_test_db"]
    collection = db["my_test_collection"]

    # 3) Insert a sample document
    doc = {"name": "test_user", "msg": "Hello MongoDB!"}
    result = collection.insert_one(doc)
    print(f"Inserted document with _id: {result.inserted_id}")

    # 4) Fetch and print all documents from the collection
    print("Documents in collection:")
    for d in collection.find():
        print(d)

if __name__ == "__main__":
    main()
