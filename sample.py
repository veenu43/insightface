import cv2
import numpy as np
import json
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import cv2

image_name = "elon.jpg"
input_image_path = "./datasets/input/" + image_name
output_image_path = "./datasets/output/" + image_name

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
known_face_encodings = []

# img = ins_get_image('joe1')
image_to_detect = cv2.imread(input_image_path)
face_elon = app.get(image_to_detect)
'''
print(type(face_elon))
print(type(face_elon[0]))
print(type(face_elon[0].get('embedding')))
print(face_elon[0].normed_embedding)
print(f"length{len(face_elon)}")
data = json.loads(face_elon[0])
print(data['embedding'])
'''
print(face_elon)
print(type(face_elon[0].get('embedding')))
print("Elon:", face_elon[0].embedding_norm)

image_name = "biden.jpg"
input_image_path = "./datasets/input/" + image_name
image_to_detect = cv2.imread(input_image_path)
faces_biden = app.get(image_to_detect)
print("Biden:", faces_biden[0].embedding_norm)

image_name = "biden2.jpg"
input_image_path = "./datasets/input/" + image_name
image_to_detect = cv2.imread(input_image_path)
faces_biden2 = app.get(image_to_detect)
print("Biden2:", faces_biden2[0].embedding_norm)
# print(face_elon[0].get('embedding') == )
image_name = "modi.jpg"
input_image_path = "./datasets/input/" + image_name
image_to_detect = cv2.imread(input_image_path)
faces_modi = app.get(image_to_detect)
print("Modi:", faces_modi[0].embedding_norm)
key = 'embedding'
# 'pose','landmark_3d_68','kps','landmark_2d_106','embedding'
known_face_encodings = [face_elon[0].get(key), faces_biden[0].get(key), faces_modi[0].get(key)]

known_face_names = ["Elon", "Biden", "Modi"]

# img = app.draw_on(image_to_detect, faces)
# cv2.imwrite(output_image_path, img)

image_name = "elon2.jpg"
input_image_path = "./datasets/input/" + image_name
image_to_detect = cv2.imread(input_image_path)
face_elon2 = app.get(image_to_detect)
print("Elon2:", face_elon2[0].embedding_norm)

print(np.linalg.norm(known_face_encodings - faces_biden2[0].get(key), axis=1))
print(list(np.linalg.norm(known_face_encodings - faces_biden2[0].get(key), axis=1) <= 0.8))


known_face_norm_encodings = [face_elon[0].normed_embedding, faces_biden[0].normed_embedding, faces_modi[0].normed_embedding]
print(np.linalg.norm(known_face_norm_encodings - faces_biden2[0].normed_embedding, axis=1))
print(list(np.linalg.norm(known_face_norm_encodings - faces_biden2[0].normed_embedding, axis=1) <= 0.9))


known_face_norm_encodings = [face_elon[0].normed_embedding, faces_biden[0].normed_embedding, faces_modi[0].normed_embedding]
print(np.linalg.norm(known_face_norm_encodings - face_elon2[0].normed_embedding, axis=1))
print(list(np.linalg.norm(known_face_norm_encodings - face_elon2[0].normed_embedding, axis=1) <= 0.9))

