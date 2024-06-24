class SearchEngine:
    def __init__(self, db):
        self.db = db

    def search(self, query):
        return self.db.similarity_search(query=query)
