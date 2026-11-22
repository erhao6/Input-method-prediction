import jieba
import torch
import config
from model import InputMethodModel
from tokenizer import JiebaTokenizer


def predict_batch(model,inputs):
    model.eval()
    with torch.no_grad():
        output=model(inputs)
    top5_index=torch.topk(output,k=5).indices
    top5_index_list=top5_index.tolist()
    return top5_index_list




def predict(text,model,tokenizer,device,seq_len):

    #tackle input
    if not text.strip():
        raise ValueError("请输入非空文本")
    indexes=tokenizer.encode(text)[-seq_len:]
    input_tensor=torch.tensor([indexes],dtype=torch.long).to(device)

    #5predict
    model.eval()
    with torch.no_grad():
        output=model(input_tensor)
    top5_index=torch.topk(output,k=5).indices
    top5_index_list=top5_index[0].tolist()
    top5_tokens=[tokenizer.index2word[index] for index in top5_index_list]
    return top5_tokens


def run_predict():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(
        config.models_dir / "best.pth", map_location=device, weights_only=True
    )
    if 'vocab' not in checkpoint or 'model_state_dict' not in checkpoint:
        raise ValueError("旧模型未包含配套词表，请重新运行 precess.py 和 train.py")
    vocab_list = checkpoint['vocab']
    tokenizer = JiebaTokenizer(vocab_list)

    # 3 model
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("欢迎使用预测模型，输入 q 退出")
    input_history=''
    while True:
        try:
            user_input=input(">")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.strip()=="q":
            break
        if user_input.strip() =='':
            print("请输入非空文本")
            continue
        input_history+=user_input
        print(f'input_history:{input_history}')
        top5_tokens=predict(input_history,model,tokenizer,device,checkpoint['seq_len'])
        print(f'top5_tokens:{top5_tokens}')
if __name__ == '__main__':
  # top5_tokens=predict("我们团队")
  # print(top5_tokens)
   run_predict()
