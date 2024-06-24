import os

class DocumentManager:
    def __init__(self, base_dir, allowed_ext):
        self.base_dir = base_dir
        self.allowed_ext = allowed_ext

    def extract_text(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def get_documents(self):
        text_data = []
        metadata = []
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                print(file)
                if file.endswith(self.allowed_ext):
                    file_path = os.path.join(root, file)
                    text_data.append(self.extract_text(file_path))
                    metadata.append({"file-name": file_path.replace(self.base_dir, "")})
        return text_data, metadata