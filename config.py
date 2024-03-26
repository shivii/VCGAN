import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


DATA_DIR_PATH = "/home/apoorvkumar/shivi/Project/data/padded/" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = DATA_DIR_PATH + "train"
VAL_DIR = DATA_DIR_PATH + "test"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 50
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_A = "genh.pth.tar"
CHECKPOINT_GEN_B = "genz.pth.tar"
CHECKPOINT_CRITIC_A = "critich.pth.tar"
CHECKPOINT_CRITIC_B = "criticz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
