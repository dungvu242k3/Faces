import os
import pickle

import torch


class FaceDatabase:
    def __init__(self, db_path="database/database.pkl"):
        self.db_path = db_path
        self.db = self.load_database()

    def load_database(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                return pickle.load(f)
        return {}  

    def save_database(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.db, f)

    def add_face(self, name, embedding, filename=None):
        record = (embedding, filename)
        if name in self.db:
            if filename and any(f == filename for _, f in self.db[name]):
                return  
            self.db[name].append(record)
        else:
            self.db[name] = [record]

    def get_all(self):
        names = []
        embeddings = []
        for name, records in self.db.items():
            for emb, _ in records:
                names.append(name)
                embeddings.append(emb)

        if len(embeddings) == 0:
            return [], torch.empty((0, 512))

        return names, torch.stack(embeddings)

    def get_filenames(self):
        return {f for records in self.db.values() for _, f in records if f}
