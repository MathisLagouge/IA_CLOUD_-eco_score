# On importe les packages
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# On recupere les donnees du dataset
raw_datasets = load_dataset("csv", data_files="data/data_int.csv")["train"]

print(raw_datasets["Sentiment"])

train, test = train_test_split(raw_datasets, test_size=0.2, shuffle=True)

raw_datasets = DatasetDict({
    "train": Dataset.from_dict({
        "Sentence": train["Sentence"],
        'Sentiment': train["Sentiment"]
    }),
    "validation": Dataset.from_dict({
        "Sentence": test["Sentence"],
        'Sentiment': test["Sentiment"]
    })
})

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# On definit la fonction pour tokenizer nos phrases
def tokenize_function(sentence):
    return tokenizer(sentence['Sentence'], truncation=True)

# On tokenise les donnees du dataset
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# On nettoie les donnees tokeniser
tokenized_datasets = tokenized_datasets.remove_columns(["Sentence"])
tokenized_datasets = tokenized_datasets.rename_column("Sentiment", "labels")
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

outputs = model(**batch)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

print("\n", device, "\n")

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

model.save_pretrained("Premier_Model")