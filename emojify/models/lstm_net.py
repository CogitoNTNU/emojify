import torch.nn as nn
import torch.nn.functional as F

# Decent article: https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0  # noqa: E501

# https://www.kaggle.com/code/arunmohan003/sentiment-analysis-using-lstm-pytorch
# This one suggest taking hte hidden state from the last layer as input to the model in
# forward as well


class LSTMNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        label_count: int,
    ):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # emd
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, label_count),
        )

    def forward(self, sentence_embeddings):
        lstm_out, *_ = self.lstm(sentence_embeddings)
        view = lstm_out.view(len(sentence_embeddings), -1)
        seq_out = self.seq(view)
        tag_scores = F.log_softmax(seq_out, dim=1)
        return tag_scores
