import os
import torch

SAVE_ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), "./checkpoints")

# GPU setting
GPU_NUMBERS = torch.cuda.device_count()
if GPU_NUMBERS > 1:
    MULTI_GPU_FLAG = True
else:
    MULTI_GPU_FLAG = False

# training datasets settings
MAX_LEN = 200
if MULTI_GPU_FLAG:
    BATCH_IMAGE_SIZE = 500000
    MAX_IMAGE_SIZE = 500000
    VALID_BATCH_SIZE = 500000
    BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
else:
    BATCH_IMAGE_SIZE = 320000
    MAX_IMAGE_SIZE = 320000
    VALID_BATCH_SIZE = 320000
    BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16

# models setting
INPUT_CHANNELS = 1
