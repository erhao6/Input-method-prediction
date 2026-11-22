import json
from pathlib import Path

import jieba
from tqdm import tqdm
import config


class JiebaTokenizer:
    unk_token = '<unk>'
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        self.unk_token_index = self.word2index[self.unk_token]

    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)
    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.word2index.get(token, self.unk_token_index) for token in tokens]


    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        vocab_set = set()
        for sentence in tqdm(sentences, desc="构建词表"):
            vocab_set.update(jieba.lcut(sentence))

        vocab_list = [cls.unk_token] + sorted(vocab_set - {cls.unk_token})
        print(f'词表大小:{len(vocab_list)}')

        # 5.保存词表
        with open(vocab_path, 'w', encoding='utf-8') as f:
            if Path(vocab_path).suffix == '.json':
                json.dump(vocab_list, f, ensure_ascii=False)
            else:
                f.write('\n'.join(vocab_list))
        return cls(vocab_list)

    @classmethod
    def from_vocab(cls, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            if Path(vocab_path).suffix == '.json':
                vocab_list = json.load(f)
            else:
                vocab_list = [line.rstrip('\r\n') for line in f]
        return cls(vocab_list)


if __name__ == '__main__':
    tokenizer = JiebaTokenizer.from_vocab(config.models_dir / 'vocab.json')
    print(f'词表大小: {tokenizer.vocab_size}')
    print(f'特殊符号: {tokenizer.unk_token}')
    print(tokenizer.encode("今天天气不错"))
