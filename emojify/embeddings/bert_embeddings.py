import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer  # type: ignore

"""
Some sources:
Keras recsys using bert: https://medium.com/analytics-vidhya/recommendation-system-using-bert-embeddings-1d8de5fc3c56 # noqa

BERT word embeddings: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
"""

BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

BERT_MODEL = BertModel.from_pretrained(
    "bert-base-uncased",
    output_hidden_states=True,
)


def get_bert_embeddings_for_texts(
    articles: list[str], max_text_length: int = 200
) -> np.ndarray:
    """Creates BERT embeddings for a list of texts where each embedding is a sentence
    embedding.
    Each text embedding is a vector of size 768.

    Args:
        articles (list[str]): List of texts to find the embeddings for.

    Returns:
        np.ndarray: An array of size (len(articles), 768) where each row is a text
        embedding vector.
    """
    texts = [text[:max_text_length] for text in articles]
    return np.array([get_bert_embeddings_full(prep) for prep in tqdm(texts)])


def bert_text_preparation(text: str):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = BERT_TOKENIZER.tokenize(marked_text)
    indexed_tokens = BERT_TOKENIZER.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens], device="cuda")
    segments_tensors = torch.tensor([segments_ids], device="cuda")

    return tokenized_text, tokens_tensor, segments_tensors


def get_bert_sentence_embedding(tokens_tensor, segments_tensors) -> np.ndarray:
    # Gradient calculation id disabled
    # Model is in inference mode
    BERT_MODEL.cuda()  # type: ignore
    with torch.no_grad():
        outputs = BERT_MODEL(
            tokens_tensor, segments_tensors, output_hidden_states=True
        )  # type: ignore
        hidden_states = outputs[2]

    # Token embeddings for each token
    # token_embeddings_stacked = torch.stack(hidden_states, dim=0)
    # token_embeddings = torch.squeeze(token_embeddings_stacked, dim=1)
    # token_embeddings_permuted = token_embeddings.permute(1, 0, 2)
    # sum_vec = [torch.sum(token[-4:], dim=0) for token in token_embeddings_permuted]

    # One token vector for the entire sentence
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    # Converting torchtensors to lists
    # list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return sentence_embedding.cpu().numpy()


def get_bert_embeddings_full(text: str):
    # Getting embeddings for the target
    # word in all given contexts

    _, tokens_tensor, segments_tensors = bert_text_preparation(text)
    list_token_embeddings = get_bert_sentence_embedding(tokens_tensor, segments_tensors)

    # Find the position 'bank' in list of tokens
    # Get the embedding for bank
    return list_token_embeddings
    # return list_token_embeddings[word_index]


def main():
    print("Reading articles")
    df1 = pd.read_csv("data/train.txt", sep=";", names=["text", "label"])
    df2 = pd.read_csv("data/val.txt", sep=";", names=["text", "label"])
    df3 = pd.read_csv("data/test.txt", sep=";", names=["text", "label"])
    df = pd.concat([df1, df2, df3])
    print("Creating embeddings")
    embeddings = get_bert_embeddings_for_texts(df["text"].values.tolist())
    print("Saving embeddings")
    np.save("data/bert_embeddings.npy", embeddings)
    np.save("data/bert_embeddings_labels.npy", df["label"].to_list())
    print("Done creating BERT embeddings")
    print("The bert embeddings")
    with open("data/bert_embeddings.npy", "rb") as f:
        em = np.load(f)
        print(em)


if __name__ == "__main__":
    main()
