import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from datasets import load_metric

from .utils import get_data, plot, announce, report_data
from .consts import category_dict
from datasets import load_metric

class DataSet(torch.utils.data.Dataset):
    """
    A custom dataset class for handling text data.
    """

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.texts.items()}
        item['labels'] = torch.tensor(self.labels[index], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def compute_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy_metric = load_metric("accuracy")
    return accuracy_metric.compute(predictions=predictions, references=labels)

def transformer_classification(portion: float = 1.0) -> float:
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base',
                                                                num_labels=len(category_dict),
                                                                problem_type="single_label_classification")

    # Define hyperparameters
    NUM_EPOCHS = 3
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 16

    # Load and tokenize data
    x_train, y_train, x_test, y_test = get_data(categories=list(category_dict.keys()), portion=portion)
    train_encodings = tokenizer(x_train, truncation=True, padding=True)
    test_encodings = tokenizer(x_test, truncation=True, padding=True)

    # Create datasets
    train_dataset = DataSet(train_encodings, y_train)
    test_dataset = DataSet(test_encodings, y_test)

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        output_dir="./output",
        logging_dir="./logs",
        load_best_model_at_end=True,
        disable_tqdm=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_accuracy,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train and evaluate
    trainer.train()
    evaluation_results = trainer.evaluate()

    return evaluation_results["eval_accuracy"]

@announce
def q2(portions: list[float]) -> None:
    print("Finetuning results:")
    transformer_validation_accuracies = []
    for p in portions:
        validation_accuracy = transformer_classification(portion=p)
        print(f"Portion: {p}")
        transformer_validation_accuracies.append(validation_accuracy)

    plot(accuracies=transformer_validation_accuracies, portions=portions, model_name="Finetuning_Transformer")


__all__ = [
    "transformer_classification",
    "q2"
]
