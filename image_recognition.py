import numpy as np
from insightface.app import FaceAnalysis
import cv2
from db.mongodb_operations import save_to_db, fetchAll
import os


app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


def get_embeddings(image_name, path):
    input_image_path = path + image_name
    image_to_detect = cv2.imread(input_image_path)
    face_embeddings = app.get(image_to_detect)
    return face_embeddings


def match_euclidean(face_embedding, known_face_encodings):
    euclidean_value = np.linalg.norm(known_face_encodings - face_embedding[0].normed_embedding, axis=1)
    print(euclidean_value)
    print(list(euclidean_value <= 1.05))


def match_cosine(face_embedding, known_face_encodings):
    cosine_value = np.dot(known_face_encodings, face_embedding[0].normed_embedding) / np.linalg.norm(
        known_face_encodings, axis=1) * np.linalg.norm(face_embedding[0].normed_embedding)
    print(cosine_value)
    print(list(cosine_value > 0.4))


def main():
    known_face_encodings = fetchAll('faceRecognition', 'insightface_buffalo_l', 'normed_embedding')
    print("known_face_encodings: ", known_face_encodings)
    known_face_encodings_landmark = []
    known_face_names = []
    for document in known_face_encodings:
        known_face_encodings_landmark.append(document['normed_embedding'])
        known_face_names.append(document['image_name'])
    print("known_face_encodings_landmark: ", known_face_encodings_landmark)
    path = "./datasets/testing/"
    images = os.listdir(path)
    for image in images:
        print("image", image)
        face_embedding = get_embeddings(image, path)
        match_cosine(face_embedding, known_face_encodings_landmark)
        print(known_face_names)


if __name__ == "__main__":
    main()
