import cv2
import numpy as np
import json
import insightface
from schema.ModelPerformanceMetric import ModelOutput, ModelPerformanceMetricEncoder
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import cv2
from db.mongodb_operations import save_to_db,fetchAll
import os
import glob

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))



def get_embeddings(image_name,path):
    input_image_path = path + image_name
    image_to_detect = cv2.imread(input_image_path)
    face_embeddings = app.get(image_to_detect)
    #print(f"{image_name} has embeddings: {face_embeddings[0].embedding_norm}")
    #print(f"{image_name}:Age{face_embeddings[0].get('age')}, Gender {face_embeddings[0].get('gender')}")
    return face_embeddings

def match_euclidean(face_embedding,known_face_encodings):
    euclidean_value= np.linalg.norm(known_face_encodings - face_embedding[0].normed_embedding, axis=1)
    print(euclidean_value)
    print(list(euclidean_value <= 1.05))

def match_cosine(face_embedding,known_face_encodings):
    cosine_value= np.dot(known_face_encodings, face_embedding[0].normed_embedding) / np.linalg.norm(
        known_face_encodings, axis=1) * np.linalg.norm(face_embedding[0].normed_embedding)
    print(cosine_value)
    print(list(cosine_value > 0.4))
    '''
    for known_face_encoding in known_face_encodings:
        print(np.dot(known_face_encoding,face_embedding[0].normed_embedding)/np.linalg.norm(known_face_encoding)*np.linalg.norm(face_embedding[0].normed_embedding))
    '''
    #print(list(np.dot(known_face_encodings - face_embedding[0].normed_embedding) <= 0.5))
def main():
    #print("face_embeddings: ",face_embedding)
    known_face_encodings = fetchAll('faceRecognition', 'insightface_buffalo_l','normed_embedding')
    print("known_face_encodings: ", known_face_encodings)
    known_face_encodings_landmark = []
    known_face_names = []
    for document in known_face_encodings:
        known_face_encodings_landmark.append(document['normed_embedding'])
        known_face_names.append(document['image_name'])
    print("known_face_encodings_landmark: ", known_face_encodings_landmark)
    path = "./datasets/testing/"
    images = os.listdir(path)
    #print("images", images)
    #print(glob.glob(path + "*.jpg"))
    for image in images:
        print("image", image)
        face_embedding = get_embeddings(image, path)
        match_cosine(face_embedding, known_face_encodings_landmark)
        print(known_face_names)



if __name__ == "__main__":
    main()

