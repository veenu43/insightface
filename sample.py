import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import cv2

image_name = "elon.jpg"
input_image_path = "./datasets/"+image_name
output_image_path = "./datasets/output/"+image_name

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
#img = ins_get_image('joe1')
image_to_detect = cv2.imread(input_image_path)
faces = app.get(image_to_detect)

rimg = app.draw_on(image_to_detect, faces)
cv2.imwrite(output_image_path, rimg)


