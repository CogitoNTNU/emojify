import torch.nn as nn
import torch.nn.functional as F

# Decent article: https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0  # noqa: E501


class ExampleNet(nn.Module):
    def __init__(
        self,
        batch_size: int,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        label_count: int,
    ):
        super(ExampleNet, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim * batch_size)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # emd
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            batch_size * embedding_dim**2, hidden_dim, 2, batch_first=True
        )
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 4, label_count)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        some = embeds.view(len(sentence), 1, -1)
        lstm_out, *_ = self.lstm(some)
        out = self.relu(self.linear1(lstm_out.view(len(sentence), -1)))
        tag_space = self.relu(self.hidden2tag(out))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
