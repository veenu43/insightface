import cv2
import numpy as np
import json
import insightface
from schema.ModelPerformanceMetric import ModelOutput, ModelPerformanceMetricEncoder
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import cv2
from db.mongodb_operations import save_to_db


def model_json(imagName, bbox, kps, landmark_3d_68, det_score, pose, landmark_2d_106, gender, age, embedding):
    return ModelOutput(imagName, bbox, kps, landmark_3d_68, det_score, pose, landmark_2d_106, gender, age, embedding)


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
performance_metric_json = json.dumps(modeloutput.__dict__)

print(type(performance_metric_json))

'''
'''
modeloutput = modeljson(self,face_elon[0].get('bbox'),face_elon[0].get('kps'),face_elon[0].get('landmark_3d_68'),face_elon[0].get('det_score'),face_elon[0].get('pose'),
                       face_elon[0].get('landmark_2d_106'),face_elon[0].get('gender')),face_elon[0].get('age')),face_elon[0].get('embedding'))
print(test("Success"))
print(type(face_elon))
print(type(face_elon[0]))
print(type(face_elon[0].get('embedding')))
print(face_elon[0].normed_embedding)
print(f"length{len(face_elon)}")
#data = json.loads(face_elon[0])
#print(data['embedding'])

'''
print(face_elon)
print(type(face_elon))
print(type(face_elon[0]))
print(type(face_elon[0].get('embedding')))
print("Elon:", face_elon[0].embedding_norm)
print(f"Elon:Age{face_elon[0].get('age')}, Gender {face_elon[0].get('gender')}")

image_name = "elon2.jpg"
input_image_path = "./datasets/input/" + image_name
image_to_detect = cv2.imread(input_image_path)
face_elon2 = app.get(image_to_detect)
print("Elon2:", face_elon2[0].embedding_norm)
print(f"Elon2:Age{face_elon2[0].get('age')}, Gender {face_elon2[0].get('gender')}")


image_name = "biden.jpg"
input_image_path = "./datasets/input/" + image_name
image_to_detect = cv2.imread(input_image_path)
faces_biden = app.get(image_to_detect)
print("Biden:", faces_biden[0].embedding_norm)
print(f"Biden:Age{faces_biden[0].get('age')}, Gender {faces_biden[0].get('gender')}")



image_name = "biden2.jpg"
input_image_path = "./datasets/input/" + image_name
image_to_detect = cv2.imread(input_image_path)
faces_biden2 = app.get(image_to_detect)
print("Biden2:", faces_biden2[0].embedding_norm)
print(f"Biden2:Age{faces_biden2[0].get('age')}, Gender {faces_biden2[0].get('gender')}")
# print(face_elon[0].get('embedding') == )



image_name = "modi.jpg"
input_image_path = "./datasets/input/" + image_name
image_to_detect = cv2.imread(input_image_path)
faces_modi = app.get(image_to_detect)
print("Modi:", faces_modi[0].embedding_norm)
print(f"Modi:Age{faces_modi[0].get('age')}, Gender {faces_modi[0].get('gender')}")


image_name = "Modi2.jpg"
input_image_path = "./datasets/input/" + image_name
image_to_detect = cv2.imread(input_image_path)
face_modi2 = app.get(image_to_detect)
print("Modi2:", face_modi2[0].embedding_norm)
print(f"Modi2:Age{face_modi2[0].get('age')}, Gender {face_modi2[0].get('gender')}")



image_name = "Modi3.jpg"
input_image_path = "./datasets/input/" + image_name
image_to_detect = cv2.imread(input_image_path)
face_modi3 = app.get(image_to_detect)
print("Modi3:", face_modi3[0].embedding_norm)
print(f"Modi3:Age{face_modi3[0].get('age')}, Gender {face_modi3[0].get('gender')}")



image_name = "Modi4.jpg"
input_image_path = "./datasets/input/" + image_name
image_to_detect = cv2.imread(input_image_path)
face_modi4 = app.get(image_to_detect)
print("Modi4:", face_modi4[0].embedding_norm)
print(f"Modi4:Age{face_modi4[0].get('age')}, Gender {face_modi4[0].get('gender')}")


image_name = "Modi5.jpg"
input_image_path = "./datasets/input/" + image_name
image_to_detect = cv2.imread(input_image_path)
face_modi5 = app.get(image_to_detect)
print("Modi5:", face_modi5[0].embedding_norm)
print(f"Modi5:Age{face_modi5[0].get('age')}, Gender {face_modi5[0].get('gender')}")


image_name = "Modi6.jpg"
input_image_path = "./datasets/input/" + image_name
image_to_detect = cv2.imread(input_image_path)
face_modi6 = app.get(image_to_detect)
print("Modi6:", face_modi6[0].embedding_norm)
print(f"Modi6:Age{face_modi6[0].get('age')}, Gender {face_modi6[0].get('gender')}")



image_name = "Modi7.jpg"
input_image_path = "./datasets/input/" + image_name
image_to_detect = cv2.imread(input_image_path)
face_modi7 = app.get(image_to_detect)
print("Modi7:", face_modi7[0].embedding_norm)
print(f"Modi7:Age{face_modi7[0].get('age')}, Gender {face_modi7[0].get('gender')}")


image_name = "Modi8.jpg"
input_image_path = "./datasets/input/" + image_name
image_to_detect = cv2.imread(input_image_path)
face_modi8 = app.get(image_to_detect)
print("Modi8:", face_modi8[0].embedding_norm)
print(f"Modi8:Age{face_modi8[0].get('age')}, Gender {face_modi8[0].get('gender')}")


image_name = "Modi9.jpg"
input_image_path = "./datasets/input/" + image_name
image_to_detect = cv2.imread(input_image_path)
face_modi9 = app.get(image_to_detect)
print("Modi9:", face_modi9[0].embedding_norm)
print(f"Modi9:Age{face_modi9[0].get('age')}, Gender {face_modi9[0].get('gender')}")



key = 'landmark_3d_68'
# 'pose','landmark_3d_68','kps','landmark_2d_106','embedding'
known_face_encodings = [face_elon[0].get(key), faces_biden[0].get(key), faces_modi[0].get(key)]

known_face_names = ["Elon", "Biden", "Modi"]

# img = app.draw_on(image_to_detect, faces)
# cv2.imwrite(output_image_path, img)


print(np.linalg.norm(known_face_encodings - faces_biden2[0].get(key), axis=1))
print(list(np.linalg.norm(known_face_encodings - faces_biden2[0].get(key), axis=1) <= 0.9))

known_face_norm_encodings = [face_elon[0].normed_embedding, faces_biden[0].normed_embedding,
                             faces_modi[0].normed_embedding]
print(np.linalg.norm(known_face_norm_encodings - faces_biden2[0].normed_embedding, axis=1))
print(list(np.linalg.norm(known_face_norm_encodings - faces_biden2[0].normed_embedding, axis=1) <= 0.9))

known_face_norm_encodings = [face_elon[0].normed_embedding, faces_biden[0].normed_embedding,
                             faces_modi[0].normed_embedding]
print(np.linalg.norm(known_face_norm_encodings - face_elon2[0].normed_embedding, axis=1))
print(list(np.linalg.norm(known_face_norm_encodings - face_elon2[0].normed_embedding, axis=1) <= 0.9))

known_face_norm_encodings = [face_elon[0].normed_embedding, faces_biden[0].normed_embedding,
                             faces_modi[0].normed_embedding]
print("Modi2", np.linalg.norm(known_face_norm_encodings - face_modi2[0].normed_embedding, axis=1))
print("Modi2", list(np.linalg.norm(known_face_norm_encodings - face_modi2[0].normed_embedding, axis=1) <= 0.9))

known_face_norm_encodings = [face_elon[0].normed_embedding, faces_biden[0].normed_embedding,
                             faces_modi[0].normed_embedding]
print("Modi3", np.linalg.norm(known_face_norm_encodings - face_modi3[0].normed_embedding, axis=1))
print("Modi3", list(np.linalg.norm(known_face_norm_encodings - face_modi3[0].normed_embedding, axis=1) <= 0.9))

known_face_norm_encodings = [face_elon[0].normed_embedding, faces_biden[0].normed_embedding,
                             faces_modi[0].normed_embedding]
print("Modi5", np.linalg.norm(known_face_norm_encodings - face_modi5[0].normed_embedding, axis=1))
print("Modi5", list(np.linalg.norm(known_face_norm_encodings - face_modi5[0].normed_embedding, axis=1) <= 1.1))

known_face_norm_encodings = [face_elon[0].normed_embedding, faces_biden[0].normed_embedding,
                             faces_modi[0].normed_embedding]
print("Modi4", np.linalg.norm(known_face_norm_encodings - face_modi4[0].normed_embedding, axis=1))
print("Modi4", list(np.linalg.norm(known_face_norm_encodings - face_modi4[0].normed_embedding, axis=1) <= 1.1))

known_face_norm_encodings = [face_elon[0].normed_embedding, faces_biden[0].normed_embedding,
                             faces_modi[0].normed_embedding]
print("Modi6", np.linalg.norm(known_face_norm_encodings - face_modi6[0].normed_embedding, axis=1))
print("Modi6", list(np.linalg.norm(known_face_norm_encodings - face_modi6[0].normed_embedding, axis=1) <= 1.1))

known_face_norm_encodings = [face_elon[0].normed_embedding, faces_biden[0].normed_embedding,
                             faces_modi[0].normed_embedding]
print("Modi7", np.linalg.norm(known_face_norm_encodings - face_modi7[0].normed_embedding, axis=1))
print("Modi7", list(np.linalg.norm(known_face_norm_encodings - face_modi7[0].normed_embedding, axis=1) <= 1.1))

known_face_norm_encodings = [face_elon[0].normed_embedding, faces_biden[0].normed_embedding,
                             faces_modi[0].normed_embedding]
print("Modi8", np.linalg.norm(known_face_norm_encodings - face_modi8[0].normed_embedding, axis=1))
print("Modi8", list(np.linalg.norm(known_face_norm_encodings - face_modi8[0].normed_embedding, axis=1) <= 1.1))

known_face_norm_encodings = [face_elon[0].normed_embedding, faces_biden[0].normed_embedding,
                             faces_modi[0].normed_embedding]
print("Modi9", np.linalg.norm(known_face_norm_encodings - face_modi9[0].normed_embedding, axis=1))
print("Modi9", list(np.linalg.norm(known_face_norm_encodings - face_modi9[0].normed_embedding, axis=1) <= 1.1))
