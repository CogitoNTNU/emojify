import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class BertEmbeddingsData(Dataset):
    def __init__(self) -> None:
        super().__init__()
        with open("data/bert_embeddings.npy", "rb") as f:
            self.embeddings: np.ndarray = np.load(f)  # type: ignore
        with open("data/bert_embeddings_labels.npy", "rb") as f:
            labels: np.ndarray = np.load(f)  # type: ignore
        self.class_to_idx = {label: i for i, label in enumerate(np.unique(labels))}
        self.classes = [self.class_to_idx[label] for label in labels]
        print("classes", self.classes)
        print("classes_to_idx", self.class_to_idx)

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.embeddings[index]), torch.tensor(self.classes[index])


def load_bert_embeddings_data(batch_size: int = 32):
    data = BertEmbeddingsData()
    print(len(data))
    train, test_val = random_split(data, [int(len(data) * 0.8), int(len(data) * 0.2)])
    val, test = random_split(
        test_val, [int(len(test_val) * 0.5), int(len(test_val) * 0.5)]
    )
    trainloader = DataLoader(train, batch_size, shuffle=True)
    valloader = DataLoader(val, batch_size, shuffle=True)
    testloader = DataLoader(test, batch_size, shuffle=True)
    return trainloader, valloader, testloader
