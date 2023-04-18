import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from emojify.text_utils import NumpyPreloadedDataset
from emojify.transformer_embeddings import get_transformer_embeddings


def prepare_sequence(seq: list[str], to_ix: dict):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def get_index_maps(input_data: pd.DataFrame, column: str):
    label_to_idx: dict[str, int] = dict()
    for _, row in input_data.iterrows():
        label = row[column]
        if label not in label_to_idx:
            label_to_idx[label] = len(label_to_idx)  # type: ignore
    return label_to_idx


def load_transformer_data(batch_size: int = 32):
    data = NumpyPreloadedDataset(
        "data/transformer_embeddings.npy", "data/transformer_embeddings_labels.npy"
    )
    train, test_val = random_split(data, [int(len(data) * 0.8), int(len(data) * 0.2)])
    val, test = random_split(
        test_val, [int(len(test_val) * 0.5), int(len(test_val) * 0.5)]
    )
    trainloader = DataLoader(train, batch_size, shuffle=True)
    valloader = DataLoader(val, batch_size, shuffle=True)
    testloader = DataLoader(test, batch_size, shuffle=True)
    return trainloader, valloader, testloader


def main():
    print("Reading articles")
    df1 = pd.read_csv("data/train.txt", sep=";", names=["text", "label"])
    df2 = pd.read_csv("data/val.txt", sep=";", names=["text", "label"])
    df3 = pd.read_csv("data/test.txt", sep=";", names=["text", "label"])
    df = pd.concat([df1, df2, df3])
    print("Creating embeddings")
    texts = df["text"].values.tolist()
    embeddings = np.array([get_transformer_embeddings(prep) for prep in tqdm(texts)])
    print("Saving embeddings")
    np.save("data/transformer_embeddings.npy", embeddings)
    np.save("data/transformer_embeddings_labels.npy", df["label"].values.tolist())
    print("Done creating Transformer embeddings")
    print("The Transformer embeddings")
    with open("data/transformer_embeddings.npy", "rb") as f:
        em = np.load(f)
        print(em)


if __name__ == "__main__":
    main()
