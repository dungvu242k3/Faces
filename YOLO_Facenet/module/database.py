import os
import pickle


class FaceDatabase:
    def __init__(self, path='database/database_yolo.pkl'):
        self.path = path
        self.data = self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_database(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.data, f)

    def add_face(self, name, embedding, filename=None):
        if name not in self.data:
            self.data[name] = []

        self.data[name].append({
            "embedding": embedding,
            "filename": filename
        })

    def get_all(self):
        return self.data

    def get_filenames(self):
        filenames = []
        for entries in self.data.values():
            if isinstance(entries, dict):
                entry = entries
                if "filename" in entry:
                    filenames.append(entry["filename"])
            elif isinstance(entries, list):
                for entry in entries:
                    if "filename" in entry:
                        filenames.append(entry["filename"])
        return filenames


