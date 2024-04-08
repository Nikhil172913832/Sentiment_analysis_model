import pymongo 
url = 'mongodb+srv://nikhilarora13832:nikhil123@cluster0.qwayyb4.mongodb.net/'
client = pymongo.MongoClient(url)

db = client['test_mongo']