import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

assert insightface.__version__>='0.3'
image_name = "elon.jpg"
input_image_path = "./datasets/input/"+image_name
output_image_path = "./datasets/output/"+image_name

parser = argparse.ArgumentParser(description='insightface app test')
# general
parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
parser.add_argument('--det-size', default=640, type=int, help='detection size')
args = parser.parse_args()

app = FaceAnalysis()
app.prepare(ctx_id=args.ctx, det_size=(args.det_size,args.det_size))

#img = ins_get_image('t1')
img = cv2.imread(input_image_path)
faces = app.get(img)
assert len(faces)==6
rimg = app.draw_on(img, faces)
cv2.imwrite(output_image_path, rimg)

# then print all-to-all face similarity
feats = []
for face in faces:
    feats.append(face.normed_embedding)
feats = np.array(feats, dtype=np.float32)
sims = np.dot(feats, feats.T)
print(sims)


