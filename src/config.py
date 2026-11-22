from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
raw_data_dir = root_dir / "train_model"/"data"/"raw_data"
processed_data_dir = root_dir / "train_model"/"data"/"processed_data"
logs_dir = root_dir / "train_model"/"logs"
models_dir = root_dir / "train_model"/"models"

seq_len=10
Batch_size = 256
log_interval = 50
Embedding_dim = 128
Hidden_size = 256
lr=0.001
Epochs = 2
