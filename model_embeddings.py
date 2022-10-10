import numpy as np
import json
from schema.ModelPerformanceMetric import ModelOutput, ModelPerformanceMetricEncoder
from insightface.app import FaceAnalysis
import cv2
from db.mongodb_operations import save_to_db
import os
import glob

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


def model_json(imagName, bbox, kps, landmark_3d_68, det_score, pose, landmark_2d_106, gender, age, embedding,
               face_output):
    return ModelOutput(imagName, bbox, kps, landmark_3d_68, det_score, pose, landmark_2d_106, gender, age, embedding,
                       face_output)


def save_report(report_to_save):
    save_to_db('faceRecognition', 'insightface_buffalo_l', report_to_save)


def get_json(image_name, face_output):
    modeloutput = model_json(image_name, face_output[0].get('bbox'), face_output[0].get('kps'),
                             face_output[0].get('landmark_3d_68'), face_output[0].get('det_score'),
                             face_output[0].get('pose'),
                             face_output[0].get('landmark_2d_106'), face_output[0].get('gender'),
                             face_output[0].get('age'), face_output[0].get('embedding'), face_output)
    # print(ModelPerformanceMetricEncoder().encode(modeloutput))
    print(type(modeloutput))
    print(ModelPerformanceMetricEncoder().encode(modeloutput))

    performance_metric_json = json.dumps(modeloutput, indent=4, cls=ModelPerformanceMetricEncoder)
    print("Storing model performance report object: ", performance_metric_json)

    performance_metric_json_decoded = json.loads(performance_metric_json)
    print(performance_metric_json_decoded)

    return performance_metric_json_decoded


def get_embeddings(image_name, path):
    input_image_path = path + image_name
    image_to_detect = cv2.imread(input_image_path)
    face_image = app.get(image_to_detect)
    print(f"{image_name} has embeddings: {face_image[0].embedding_norm}")
    print(f"{image_name}:Age{face_image[0].get('age')}, Gender {face_image[0].get('gender')}")
    performance_metric_json_decoded = get_json(image_name, face_image)
    return performance_metric_json_decoded


def convert(one,two):
    it1 = iter(one)
    it2 = iter(two)
    res_dct = dict(zip(it1, it2))
    return res_dct


def get_embeddings_store(images, path):
    image_embeddings = []
    for image in images:
        image_embedding = get_embeddings(image, path)
        image_embeddings.append(image_embedding)
        print(type(image_embedding))
    print(image_embeddings)
    save_report(image_embeddings)


def main():
    path = "./datasets/input/"
    images = os.listdir(path)
    batch_size = 2
    length = len(images)
    if length > batch_size:
        split_size = (length / batch_size + 1)
    else:
        split_size = length
    splits_image_arrays = np.array_split(images, split_size)
    print("images", images)
    print(glob.glob(path + "*.jpg"))
    for splits_image_array in splits_image_arrays:
        get_embeddings_store(splits_image_array, path)


if __name__ == "__main__":
    main()
