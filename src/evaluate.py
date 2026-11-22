import hashlib
import json
import torch
import config
from model import InputMethodModel
from dataset import get_dataloader
from predict import predict_batch




def evaluate(model, test_dataloader, device):
    top1_acc_count=0
    top5_acc_count=0
    total_count=0
    for inputs,targets in test_dataloader:
        inputs = inputs.to(device)
        targets = targets.tolist()
        total_count += len(targets)
        top5_indexs_list=predict_batch(model, inputs)
        for target,top5_indexs in zip(targets,top5_indexs_list):
            if target==top5_indexs[0]:
                top1_acc_count+=1
            if target in top5_indexs:
                top5_acc_count+=1
    if total_count == 0:
        raise ValueError("测试集为空，无法计算准确率")
    return top1_acc_count/total_count,top5_acc_count/total_count




def run_evaluate():
    # 准备资源
    # 1. 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    # 2. 从模型文件读取训练时使用的词表
    checkpoint = torch.load(config.models_dir / 'best.pth',
                            map_location=device, weights_only=True)
    vocab_list = checkpoint['vocab']
    metadata = json.loads((config.processed_data_dir / 'metadata.json').read_text(encoding='utf-8'))
    vocab_hash = hashlib.sha256(json.dumps(vocab_list, ensure_ascii=False).encode('utf-8')).hexdigest()
    if vocab_hash != metadata['vocab_sha256'] or checkpoint['seq_len'] != metadata['seq_len']:
        raise ValueError("测试集与模型的词表或序列长度不匹配")
    print("词表加载成功")



    # 3. 模型
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("模型加载成功")

    # 4. 数据集
    test_dataloader = get_dataloader(train=False)

    top1_acc,top5_acc=evaluate(model, test_dataloader, device)

    print(f"测试样本数：{len(test_dataloader.dataset)}")
    print(f"Top-1 准确率：{top1_acc:.2%}")
    print(f"Top-5 准确率：{top5_acc:.2%}")


if __name__ == "__main__":
    run_evaluate()

