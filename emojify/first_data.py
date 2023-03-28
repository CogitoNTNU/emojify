import os

import pandas as pd  # type: ignore
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def prepare_sequence(seq: list[str], to_ix: dict):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


class SentimentDataset(Dataset):
    def __init__(self, folder: str, max_sentence_length: int = 200):
        # Move into transforms (classes with __call__), and use transforms.Compose and
        # load that to dataloader
        input_data = pd.DataFrame()
        for file in os.listdir(folder):
            data = pd.read_csv(f"{folder}/{file}", sep=";", names=["text", "label"])
            input_data = pd.concat([input_data, data], ignore_index=True)
        input_data["text"] = input_data["text"].str.split()  # type: ignore
        self.word_to_idx, self.label_to_idx = self._get_index_maps(input_data)
        self.data = pd.DataFrame(
            {
                "text": input_data["text"].apply(
                    lambda x: prepare_sequence(x[:max_sentence_length] + [""] * (max_sentence_length - len(x)), self.word_to_idx)  # type: ignore # noqa
                ),
                "label": input_data["label"].apply(
                    lambda x: prepare_sequence([x], self.label_to_idx)[0]  # type: ignore # noqa
                ),
            }
        )
        # print("data", self.data)
        self.vocabulary_size = len(self.word_to_idx)
        self.label_count = len(self.label_to_idx)

    @staticmethod
    def _get_index_maps(input_data: pd.DataFrame):
        label_to_idx: dict[str, int] = dict()
        word_to_idx: dict[str, int] = {"": 0}
        for _, row in input_data.iterrows():
            text = row["text"]
            label = row["label"]
            if label not in label_to_idx:
                label_to_idx[label] = len(label_to_idx)  # type: ignore
            for word in text:
                if word not in word_to_idx:  # word has not been assigned an index yet
                    word_to_idx[word] = len(
                        word_to_idx
                    )  # Assign each word with a unique index
        return word_to_idx, label_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]  # type: ignore
        label = row["label"]  # type: ignore
        return text, label


def load_sentiment_data(batch_size: int = 32, max_sentence_length: int = 200):
    data = SentimentDataset("./data", max_sentence_length)
    print(len(data))
    train, test_val = random_split(data, [int(len(data) * 0.8), int(len(data) * 0.2)])
    val, test = random_split(
        test_val, [int(len(test_val) * 0.5), int(len(test_val) * 0.5)]
    )
    trainloader = DataLoader(train, batch_size, shuffle=True)
    valloader = DataLoader(val, batch_size, shuffle=True)
    testloader = DataLoader(test, batch_size, shuffle=True)
    return trainloader, valloader, testloader, data.word_to_idx, data.label_to_idx
