from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextSplitter:
    def __init__(self, chunk_size, chunk_overlap):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split(self, text_data, metadata):
        docs = self.splitter.create_documents(text_data, metadata)
        return self.splitter.split_documents(docs)