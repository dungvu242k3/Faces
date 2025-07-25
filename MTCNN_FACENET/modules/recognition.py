import torch


class FaceRecognizer:
    def __init__(self, database, threshold=0.8, device="cpu"):
        self.db = database
        self.threshold = threshold
        self.device = device
        self.names, self.embeddings = self.db.get_all()
        self.embeddings = self.embeddings.to(self.device)

    def recognize(self, embedding):
        if len(self.embeddings) == 0:
            return "Unknown", None, None

        diff = self.embeddings - embedding.unsqueeze(0)
        dists = torch.norm(diff, dim=1)

        min_dist, min_idx = torch.min(dists, dim=0)

        if min_dist.item() < self.threshold:
            return self.names[min_idx], min_dist.item(), min_idx.item()
        else:
            return "Unknown", min_dist.item(), None
