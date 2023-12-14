from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_dataset


import os

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)


dataset = load_dataset("md_gender_bias", name="convai2_inferred")

# Access the train, validation, and test splits
train_data = Dataset.from_dict(dataset["train"][0:5000])

validation_data = dataset["validation"]
test_data = dataset["test"]

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def process_dataset(dataset):
    new_dataset_dict = {
        'text': [entry['text'] for entry in dataset],
        'labels': [entry['binary_score'] if entry['binary_label'] == 1 else 1 - entry['binary_score'] for entry in dataset]
    }
    new_dataset = Dataset.from_dict(new_dataset_dict)
    return new_dataset


training_args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_total_limit=2,
    evaluation_strategy="epoch",
)

processed_train_dataset = process_dataset(train_data)
processed_validation_dataset = process_dataset(validation_data)
tokenized_train_datasets = processed_train_dataset.map(tokenize_function, batched=True)
tokenized_validation_datasets = processed_validation_dataset.map(tokenize_function, batched=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_validation_datasets,
    tokenizer=tokenizer,
    compute_metrics=None,  # Replace with your metric function for regression if needed
)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets,  # Use combined policy and non-policy inputs
# )

trainer.train()

model.save_pretrained('model_saved')