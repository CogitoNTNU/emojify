import torch.nn as nn
import torch.nn.functional as F

from emojify.first_data import load_sentiment_data
from emojify.nn_manager import NNManager

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


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    # utils.set_seed(0)
    embedding_dim = 32  # should probably be 32
    hidden_dim = 6  # should probably be 32
    epochs = 10
    batch_size = 32
    learning_rate = 5e-2
    early_stop_count = 4
    train, val, test, word_to_idx, label_to_idx = load_sentiment_data(
        batch_size, embedding_dim
    )
    model = ExampleNet(
        batch_size,
        embedding_dim,
        hidden_dim,
        vocab_size=len(word_to_idx),
        label_count=len(label_to_idx),
    )
    nn_manager = NNManager(
        batch_size, learning_rate, early_stop_count, epochs, model, (train, val, test)
    )
    nn_manager.train()
    acc = nn_manager.test()
    print(f"Test accuracy: {acc:.2f}")


if __name__ == "__main__":
    main()
