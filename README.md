# Input method prediction

基于 Jieba 分词和 PyTorch RNN 的中文下一个词预测示例。仓库包含源码、原始数据、词表和已训练模型。训练集与测试集由原始数据生成，因此不重复上传生成文件。

## 环境准备

建议使用 Python 3.12：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 直接预测

```bash
python src/predict.py
```

输入文字后显示前五个候选词；输入 `q` 退出。默认读取 `models/best.pth`，其中已包含训练时的词表。

## 重新生成数据、训练和评估

```bash
python src/precess.py
python src/train.py
python src/evaluate.py
```

原始数据为 `data/raw_data/synthesized_.jsonl`。预处理会生成 `data/processed_data/train.jsonl`、`test.jsonl` 和 `metadata.json`，并更新 `models/vocab.json`。**重新预处理后必须重新训练**，否则模型的词编号可能与新数据不匹配。训练轮数、批量大小和学习率在 `src/config.py` 中配置。

训练日志位于 `logs/`，可用 TensorBoard 查看：

```bash
python -m tensorboard.main --logdir logs
```
