import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def get_index_maps(input_data: pd.DataFrame, column: str):
    label_to_idx: dict[str, int] = dict()
    for _, row in input_data.iterrows():
        label = row[column]
        if label not in label_to_idx:
            label_to_idx[label] = len(label_to_idx)  # type: ignore
    return label_to_idx


class NumpyPreloadedDataset(Dataset):
    def __init__(self, embeddings_file: str, labels_file: str) -> None:
        super().__init__()
        with open(embeddings_file, "rb") as f:
            self.embeddings: np.ndarray = np.load(f)  # type: ignore
        with open(labels_file, "rb") as f:
            labels: np.ndarray = np.load(f)  # type: ignore
        self.class_to_idx = {label: i for i, label in enumerate(np.unique(labels))}
        self.classes = [self.class_to_idx[label] for label in labels]
        print("classes", self.classes)
        print("classes_to_idx", self.class_to_idx)

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.embeddings[index]), torch.tensor(self.classes[index])
