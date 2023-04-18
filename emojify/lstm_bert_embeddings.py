import torch
import torch.nn as nn

from emojify.bert_embedded_data import load_bert_embeddings_data
from emojify.nn_manager import NNManager

# Decent article: https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0  # noqa: E501

# https://www.kaggle.com/code/arunmohan003/sentiment-analysis-using-lstm-pytorch
# This one suggest taking hte hidden state from the last layer as input to the model in
# forward as well


class ExampleNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        batch_size: int,
        dropout=0.2,
    ):
        super().__init__()

        # The embedding layer takes the vocab size and the embeddings size as input
        # The embeddings size is up to you to decide, but common sizes are between 50 and 100.

        # The LSTM layer takes in the the embedding size and the hidden vector size.
        # The hidden dimension is up to you to decide, but common values are 32, 64, 128
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.batch_size = batch_size

        # We use dropout before the final layer to improve with regularization
        self.dropout = nn.Dropout(dropout)

        # The fully-connected layer takes in the hidden dim of the LSTM and
        #  outputs a a 3x1 vector of the class scores.
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, embedding, hidden):
        """
        The forward method takes in the input and the previous hidden state
        """

        # The input is transformed to embeddings by passing it to the embedding layer

        # The embedded inputs are fed to the LSTM alongside the previous hidden state
        out, hidden = self.lstm(embedding, hidden)

        # Dropout is applied to the output and fed to the FC layer
        out = self.dropout(out)
        out = self.fc(out)

        # We extract the scores for the final hidden state since it is the one that matters.
        out = out[:, -1]
        return out, hidden

    def init_hidden(self):
        return (
            torch.zeros(1, self.batch_size, 32),
            torch.zeros(1, self.batch_size, 32),
        )


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    # utils.set_seed(0)
    embedding_dim = 768  # should probably be 32
    hidden_dim = 6  # should probably be 32
    epochs = 10
    batch_size = 32
    learning_rate = 5e-3
    early_stop_count = 4
    train, val, test = load_bert_embeddings_data(batch_size)
    model = ExampleNet(embedding_dim, hidden_dim, 6)
    nn_manager = NNManager(
        batch_size, learning_rate, early_stop_count, epochs, model, (train, val, test)
    )
    nn_manager.train()
    acc = nn_manager.test()
    print(f"Test accuracy: {acc:.2f}")


if __name__ == "__main__":
    main()
