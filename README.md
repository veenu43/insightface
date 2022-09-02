# Python environment
conda create --name python_3.9-env-insightface python=3.9
activate python_3.9-env-insightface

# Install library
pip install -U Cython cmake numpy
pip install -U insightface
pip install onnxruntime-gpu==1.11.0
