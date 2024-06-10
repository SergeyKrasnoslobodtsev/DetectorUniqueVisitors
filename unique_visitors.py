from scipy.spatial import distance

class UniqueVisitors:
    def __init__(self):
        self.unique_faces = []
        self.unique_ids = []
        self.counter = 0

    def is_unique(self, descriptor, threshold=0.8):
        for unique_descriptor in self.unique_faces:
            if distance.euclidean(descriptor, unique_descriptor) < threshold:
                return False
        return True

    def add_face(self, descriptor):
        self.unique_faces.append(descriptor)
        self.counter += 1
        self.unique_ids.append(self.counter)

    def count(self):
        return len(self.unique_faces)

    def get_id(self, descriptor, threshold=0.45):
        for idx, unique_descriptor in enumerate(self.unique_faces):
            if distance.euclidean(descriptor, unique_descriptor) < threshold:
                return self.unique_ids[idx]
        return None