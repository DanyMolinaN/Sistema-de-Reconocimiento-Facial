import numpy as np
from typing import List, Tuple

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class Track:
    def __init__(self, track_id, bbox, embedding):
        self.track_id = track_id
        self.bbox = bbox  # x,y,w,h
        self.embedding = embedding
        self.age = 0
        self.hits = 1

    def is_confirmed(self):
        return self.hits >= 2

    def to_ltrb(self):
        x, y, w, h = self.bbox
        return (x, y, x + w, y + h)

class DeepSort:
    def __init__(self, max_age=30, appearance_threshold=0.35):
        self.max_age = max_age
        self.appearance_threshold = appearance_threshold
        self.tracks = {}
        self.next_id = 1

    def update_tracks(
        self,
        detections: List[Tuple[Tuple[int,int,int,int], np.ndarray]],
    ):
        used_tracks = set()
        updated = []

        for bbox, emb in detections:
            best_id = None
            best_dist = float("inf")

            for tid, tr in self.tracks.items():
                dist = cosine_distance(emb, tr.embedding)
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None and best_dist < self.appearance_threshold:
                tr = self.tracks[best_id]
                tr.bbox = bbox
                tr.embedding = emb
                tr.age = 0
                tr.hits += 1
                updated.append(tr)
                used_tracks.add(best_id)
            else:
                tr = Track(self.next_id, bbox, emb)
                self.tracks[self.next_id] = tr
                updated.append(tr)
                self.next_id += 1

        # Envejecer tracks no usados
        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                self.tracks[tid].age += 1
                if self.tracks[tid].age > self.max_age:
                    del self.tracks[tid]

        return updated
