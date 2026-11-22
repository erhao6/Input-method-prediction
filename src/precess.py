import hashlib
import json
from tokenizer import JiebaTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import config


def build_dataset(sentences, tokenizer):
    dataset = []
    for text in sentences:
        sentence = tokenizer.encode(text)
        for i in range(len(sentence) - config.seq_len):
            input = sentence[i:i + config.seq_len]
            target = sentence[i + config.seq_len]
            dataset.append({'input': input, 'target': target})
    return dataset


def precess():
    print("Precessing data...")
    df = pd.read_json(
        config.raw_data_dir/"synthesized_.jsonl",
        lines=True,
        orient="records"
                      )
    print(df.head())
    print("Data done")
    sentences=[]
    for dialog in df["dialog"]:
        for sentence in dialog:
            sentences.append(sentence.split("：", 1)[1])
    #print(sentences[0:10])
    print({len(sentences)} )
    #3split
    train_sentences,test_sentences=train_test_split(sentences,test_size=0.2,random_state=42)
    # 先创建并保存词表，再用同一份编号生成数据。
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.processed_data_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = JiebaTokenizer.build_vocab(train_sentences, config.models_dir / 'vocab.json')
    vocab_list = tokenizer.vocab_list
    vocab_json = json.dumps(vocab_list, ensure_ascii=False)
    (config.models_dir / 'vocab.txt').write_text("\n".join(vocab_list), encoding='utf-8')
    train_dataset=build_dataset(train_sentences,tokenizer)

    #7save
    pd.DataFrame(train_dataset).to_json(config.processed_data_dir/"train.jsonl",orient="records",lines=True)
    train_count = len(train_dataset)
    del train_dataset
    #8
    test_dataset=build_dataset(test_sentences,tokenizer)
    #9
    pd.DataFrame(test_dataset).to_json(config.processed_data_dir/"test.jsonl",orient="records",lines=True)
    metadata = {
        'vocab_sha256': hashlib.sha256(vocab_json.encode('utf-8')).hexdigest(),
        'seq_len': config.seq_len,
        'train_samples': train_count,
        'test_samples': len(test_dataset),
    }
    (config.processed_data_dir / 'metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8'
    )
    print(f"Saved {len(vocab_list)} tokens, {train_count} training samples, "
          f"{len(test_dataset)} test samples")
if __name__ == "__main__":
    precess()
