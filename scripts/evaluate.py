import torch
from transformers import RobertaForTokenClassification, RobertaTokenizer
from datasets import load_dataset
from utils import get_device, tokenize_and_align_labels, create_data_loader
from seqeval.metrics import classification_report

# Parameters
model_dir = './models'
batch_size = 16

# Load Model and Tokenizer
model = RobertaForTokenClassification.from_pretrained(model_dir)
tokenizer = RobertaTokenizer.from_pretrained(model_dir)

# Device
device = get_device()
model.to(device)

# Load Dataset
dataset = load_dataset('conll2003')
test_dataset = dataset['test']

# Tokenize Data
test_dataset = test_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# DataLoader
test_loader = create_data_loader(test_dataset, batch_size, SequentialSampler)

# Evaluation Function
def evaluate(model, data_loader, device):
    model.eval()
    true_labels = []
    pred_labels = []

    label_list = test_dataset.features['ner_tags'].feature.names

    with torch.no_grad():
        for batch in data_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_attention_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            outputs = model(
                b_input_ids,
                attention_mask=b_attention_mask
            )
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)
            for i, label in enumerate(b_labels):
                true_labels.append([label_list[l] for l in label if l != -100])
                pred_labels.append([label_list[p] for p, l in zip(predictions[i], label) if l != -100])

    return classification_report(true_labels, pred_labels)

# Evaluate
report = evaluate(model, test_loader, device)
print(report)