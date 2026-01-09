from src.dataset import load_data
from src.model import build_model
from src.train import train_model

DATA_DIR = "data/chest_xray"

train_gen, val_gen, test_gen = load_data(DATA_DIR)
model = build_model()
train_model(model, train_gen, val_gen, epochs=25)
