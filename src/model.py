from torch import nn
import config


class InputMethodModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
        embedding_dim=config.Embedding_dim
                                      )
        self.rnn=nn.RNN(input_size=config.Embedding_dim,
                        hidden_size=config.Hidden_size,
                        batch_first=True)

        self.linear = nn.Linear(in_features=config.Hidden_size,
                                out_features=vocab_size
                                )

    def forward(self,x):
        embed=self.embedding(x)
        output,hn=self.rnn(embed)
        last_hidden_state=output[:,-1,:]
        output=self.linear(last_hidden_state)
        return output
