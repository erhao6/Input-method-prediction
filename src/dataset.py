from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import config
class MethodInputDataset(Dataset):

    def __init__(self, path):
        self.data=pd.read_json(path,lines=True,orient="records").to_dict(orient="records")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        input_tensor=torch.tensor(self.data[idx]['input'],dtype=torch.long)
        target_tensor=torch.tensor(self.data[idx]['target'],dtype=torch.long)
        return input_tensor,target_tensor


def get_dataloader(train=True):
    path=config.processed_data_dir/("train.jsonl" if train else "test.jsonl")
    dataset=MethodInputDataset(path)
    return DataLoader(dataset,batch_size=config.Batch_size,shuffle=train)

if __name__=="__main__":
    train_dataloader=get_dataloader()
    test_dataloader=get_dataloader(train=False)
    print(train_dataloader)
    print(test_dataloader)
    input_tensor, target_tensor = next(iter(train_dataloader))
    print(input_tensor.shape, target_tensor.shape)
