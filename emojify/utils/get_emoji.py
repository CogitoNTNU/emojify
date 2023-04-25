import torch

import emojify.utils.nn_utils as utils
from emojify.embeddings.transformer_embeddings import get_transformer_embeddings
from emojify.models.lstm_net import LSTMNet

# class_to_idx = {"anger": 0, "fear": 1, "joy": 2, "love": 3, "sadness": 4, "surprise": 5}
emoji_to_idx = {"😠": 0, "😨": 1, "😂": 2, "😍": 3, "😢": 4, "😮": 5}
idx_to_emoji = {v: k for k, v in emoji_to_idx.items()}
state_dict = torch.load("best_checkpoints/best_model.ckpt")
embedding_dim = 384  # should probably be 32
hidden_dim = 32  # should probably be 32
this_model = LSTMNet(embedding_dim, hidden_dim, 6)
this_model.load_state_dict(state_dict)


def get_emoji(sentence):
    this_model.eval()
    with torch.no_grad():
        sentence = get_transformer_embeddings(sentence)
        input_ = torch.tensor(sentence).reshape(1, -1).to("cpu")
        tag_scores = this_model(input_)
        predicted = tag_scores.argmax(dim=1)
        return idx_to_emoji[predicted.item()]


def main():
    em = get_emoji("I am so happy")
    print(em)


if __name__ == "__main__":
    main()
