import calendar
import datetime
import time
from datetime import date, datetime
from json import JSONEncoder


def generate_unique_id():
    return calendar.timegm(time.gmtime())


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


class ModelPerformanceMetric:
    def __init__(self, trace_id, span_id, timestamp, status, model_backend, recognition_model, model_metrics,
                 accuracy_metric):
        self.id = generate_unique_id()
        self.trace_id = trace_id
        self.span_id = span_id
        self.timestamp = timestamp
        self.status = status
        self.model_backend = model_backend
        self.recognition_model = recognition_model
        self.model_metrics = model_metrics
        self.accuracy_metric = accuracy_metric


class ModelOutput:
    def __init__(self, image_name, bbox, kps, landmark_3d_68, det_score, pose, landmark_2d_106, gender, age, embedding,face_output):
        self.id = generate_unique_id()
        print("id: ", type(self.id))
        self.image_name = image_name
        print("image_name: ", type(self.image_name))
        self.bbox =  face_output[0].get('bbox').tolist()
        print("bbox: ",type(self.bbox))
        self.kps = face_output[0].get('kps').tolist()
        print("kps: ",type(self.kps))
        self.landmark_3d_68 = face_output[0].get('landmark_3d_68').tolist()
        print("landmark_3d_68: ",type(self.landmark_3d_68))
        self.det_score = float(face_output[0].get('det_score'))
        print("det_score: ", type(self.det_score))
        self.pose =  face_output[0].get('pose').tolist()
        print("pose: ",type(self.pose))
        self.landmark_2d_106 = face_output[0].get('landmark_2d_106').tolist()
        print("landmark_2d_106: ",type(self.landmark_2d_106))
        self.gender = int(face_output[0].get('gender'))
        print("gender: ", type(self.gender))
        self.age = face_output[0].get('age')
        print("age: ", type(self.age))
        self.embedding = face_output[0].get('embedding').tolist()
        print("embedding: ",type(self.embedding))
        self.normed_embedding = face_output[0].normed_embedding.tolist()
        print("normed_embedding: ", type(self.normed_embedding))


class Accuracy:
    def __init__(self, match_rate, metric_type, metric_value, threshold):
        self.match_rate = match_rate
        self.metric_type = metric_type
        self.metric_value = metric_value
        self.threshold = threshold


# subclass JSONEncoder
class ModelPerformanceMetricEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__
