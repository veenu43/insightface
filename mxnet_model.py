import pickle
import mxnet as mx
from mxnet import ndarray as nd
import os
import argparse

def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    for flip in [0, 1]:
        data = nd.empty(
            (len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][i][:] = img
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return (data_list, issame_list)

'''
image_size = [112, 112]
ver_list = []
ver_name_list = []
nets = []
data_set = load_bin("./datasets/faces_webface_112x112/cfp_ff.bin", image_size)
ver_list.append(data_set)
ver_name_list.append('cfp_ff.bin')
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='do verification')
    # general
    parser.add_argument('--data-dir', default='', help='')
    parser.add_argument('--model',
                        default='../model/softmax,50',
                        help='path to load model.')
    parser.add_argument('--target',
                        default='lfw,cfp_ff,cfp_fp,agedb_30',
                        help='test targets.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch-size', default=32, type=int, help='')
    parser.add_argument('--max', default='', type=str, help='')
    parser.add_argument('--mode', default=0, type=int, help='')
    parser.add_argument('--nfolds', default=10, type=int, help='')
    args = parser.parse_args()
    image_size = [112, 112]
    print('image_size', image_size)
    ctx = mx.gpu(args.gpu)
    nets = []
    vec = args.model.split(',')
    prefix = args.model.split(',')[0]
    epochs = []
    if len(vec) == 1:
        pdir = os.path.dirname(prefix)
        for fname in os.listdir(pdir):
            if not fname.endswith('.params'):
                continue
            _file = os.path.join(pdir, fname)
            if _file.startswith(prefix):
                epoch = int(fname.split('.')[0].split('-')[1])
                epochs.append(epoch)
        epochs = sorted(epochs, reverse=True)
        if len(args.max) > 0:
            _max = [int(x) for x in args.max.split(',')]
            assert len(_max) == 2
            if len(epochs) > _max[1]:
                epochs = epochs[_max[0]:_max[1]]