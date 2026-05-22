import json
import os

from torch.utils.data import DataLoader, Dataset


class TextPromptDataset(Dataset):
    def __init__(self, dataset_path, split="train", include_index=False):
        self.include_index = include_index
        self.file_path = os.path.join(dataset_path, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        sample = {"prompt": self.prompts[idx], "metadata": {}}
        if self.include_index:
            sample["index"] = idx
        return sample

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset_path, split="train", include_index=False):
        self.include_index = include_index
        self.file_path = os.path.join(dataset_path, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        sample = {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}
        if self.include_index:
            sample["index"] = idx
        return sample

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


def collate_examples(examples):
    return examples


def build_prompt_dataset(prompt_fn, dataset_path, split="train", include_index=False):
    if prompt_fn == "general_ocr":
        return TextPromptDataset(dataset_path, split=split, include_index=include_index)
    if prompt_fn == "geneval":
        return GenevalPromptDataset(dataset_path, split=split, include_index=include_index)
    raise ValueError(f"Unsupported prompt_fn: {prompt_fn}")


def build_eval_dataloader(prompt_fn, dataset_path, batch_size):
    dataset = build_prompt_dataset(prompt_fn, dataset_path, split="test", include_index=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=False,
        num_workers=8,
    )
