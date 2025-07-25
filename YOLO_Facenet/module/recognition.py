import numpy as np


class FaceRecognizer:
    def __init__(self, db, threshold=0.7):
        self.db = db
        self.threshold = threshold

    def recognize(self, embedding):
        min_dist = float('inf')
        identity = "Unknown"

        for name, entries in self.db.items():  
            for entry in entries:
                db_embed = entry["embedding"]
                dist = np.linalg.norm(embedding - db_embed)
                if dist < min_dist and dist < self.threshold:
                    min_dist = dist
                    identity = name

        return identity, min_dist
