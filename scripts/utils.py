# utils.py script
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizer

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        padding='max_length'
    )
    labels = []
    for i, label in enumerate(examples[f'ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def create_data_loader(dataset, batch_size, sampler):
    data_sampler = sampler(dataset)
    data_loader = DataLoader(dataset, sampler=data_sampler, batch_size=batch_size)
    return data_loader
