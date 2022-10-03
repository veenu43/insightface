# Python environment
conda create --name python_3.9-env-insightface-2 python=3.9
activate python_3.9-env-insightface-2

# Install library
pip install -U Cython cmake numpy
pip install -U insightface

pip install cuda-python
conda install -c nvidia cuda-python
pip install onnxruntime-gpu==1.11.0

pip install common
pip install dual
pip install tight
pip install data
pip install prox

pip install paddle
pip install paddlepaddle
pip install numba
