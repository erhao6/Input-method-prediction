import hashlib
import json
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import get_dataloader
from model import InputMethodModel
import config


def train_one_epoch(dataloader, model, optimizer, device,loss_fn):
    model.train()
    total_loss = torch.zeros((), device=device)
    num_batches = len(dataloader)
    progress = tqdm(dataloader, desc="训练", dynamic_ncols=True, mininterval=0.5)
    for step, (input, target) in enumerate(progress, start=1):
        inputs=input.to(device)
        targets=target.to(device)
        outputs=model(inputs)
        loss=loss_fn(outputs,targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # 在设备上累计损失，避免每一批都读取 GPU 数值。
        total_loss += loss.detach()
        if step % config.log_interval == 0 or step == num_batches:
            progress.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
    return total_loss.item()/num_batches


def Train():

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    vocab_json = (config.models_dir / 'vocab.json').read_text(encoding='utf-8')
    metadata = json.loads((config.processed_data_dir / 'metadata.json').read_text(encoding='utf-8'))
    if (hashlib.sha256(vocab_json.encode('utf-8')).hexdigest() != metadata['vocab_sha256']
            or config.seq_len != metadata['seq_len']):
        raise ValueError("训练数据与词表或序列长度不匹配，请先重新运行 precess.py")

    vocab_list = json.loads(vocab_json)
    dataloader = get_dataloader()

    #4
    model= InputMethodModel(vocab_size=len(vocab_list)).to(device)
   #5
    loss_fn=torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=config.lr)

    writer = SummaryWriter(log_dir=config.logs_dir/time.strftime('%Y%m%d_%H%M%S'))
    best_loss = float("inf")
   #
    try:
        for epoch in range(1,1 + config.Epochs):
            print("Epoch:",epoch)
            loss = train_one_epoch(dataloader,model,optimizer,device,loss_fn)
            print(f"loss:{loss}")
            writer.add_scalar("loss",loss,epoch)

            if loss < best_loss:
                best_loss = loss
                # 参数与词表一起保存，避免预测时使用另一份词编号。
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'vocab': vocab_list,
                    'seq_len': config.seq_len,
                    'epoch': epoch,
                    'train_loss': loss,
                }, config.models_dir / 'best.pth')
                print("Saved best.pth with matching vocabulary")
    finally:
        writer.close()

if __name__=="__main__":

    Train()

