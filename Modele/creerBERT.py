#Import
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
from transformers import BertForSequenceClassification
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

#Train a model for sentiment analysis
#from a sentence, predict sentiment (negative, neutral, positive)
#data format : 2 column with label "Review" type string and "Rating" type int {0,1,2}
#return : save the trained model in the save_path, print stat of tested model
#batch_size : 
#num_epochs : number of epochs for train the model
#test_size : proportion for test and train
#data_path : path for the dataset, as csv file
#save_path : path for saving the trained model
#num_label
def train_bert(batch_size,num_epochs,test_size,data_path, save_path, num_label = 3):

    # Get data from the dataset
    raw_datasets = load_dataset("csv", data_files=data_path)["train"]

    train, test = train_test_split(raw_datasets, test_size=test_size, shuffle=True)

    test = Dataset.from_dict({
        "Review": test["Review"],
        'Rating': test["Rating"]
    })

    test, validation = train_test_split(test, test_size=0.01, shuffle=True) 

    raw_datasets = DatasetDict({
        "train": Dataset.from_dict({
            "Sentence": train["Review"],
            'Sentiment': train["Rating"]
        }),
        "validation": Dataset.from_dict({
            "Sentence": validation["Review"],
            'Sentiment': validation["Rating"]
        }),
        "test": Dataset.from_dict({
            "Sentence": test["Review"],
            'Sentiment': test["Rating"]
        })
    }) 

    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # def tokenize function 
    def tokenize_function(sentence):
        return tokenizer(sentence['Sentence'], truncation=True)

    # Tokenized data from the dataset
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Clean Tokenized data
    tokenized_datasets = tokenized_datasets.remove_columns(["Sentence"])
    tokenized_datasets = tokenized_datasets.rename_column("Sentiment", "labels")
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=batch_size, collate_fn=data_collator
    )

    for batch in train_dataloader:
        break
    {k: v.shape for k, v in batch.items()}

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_label)

    outputs = model(**batch)

    optimizer = AdamW(model.parameters(), lr=5e-5)

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

    print("\nModel trained !\n")
    print("\nModel saving...\n")

    model.save_pretrained(save_path)

    print("\nModel saved !\n")
    print("\nTesting model...\n")

    #load the model
    model = BertForSequenceClassification.from_pretrained(save_path)

    tokens = tokenized_datasets['test']

    negative = 0
    neutral = 0
    positive = 0
    accuracy = 0

    #test
    for i in range(len(tokens)):

        input_ids = torch.tensor(tokens[i]['input_ids']).unsqueeze(0)
        attention_mask = torch.tensor(tokens[i]['attention_mask']).unsqueeze(0)

        outputs = model(input_ids, attention_mask=attention_mask)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        if probabilities[0][0] == max(probabilities[0]):
            negative += 1
            if(tokens[i]["labels"] == 0):
                accuracy += 1
        if probabilities[0][1] == max(probabilities[0]):
            neutral += 1
            if(tokens[i]["labels"] == 1):
                accuracy += 1
        if probabilities[0][2] == max(probabilities[0]):
            positive += 1
            if(tokens[i]["labels"] == 2):
                accuracy += 1

    print("\nModel tested !\n")

    print("Accuracy: ", accuracy/len(tokens))
    print("negative, classe 0: ", negative/len(tokens))
    print("neutral, classe 1: ", neutral/len(tokens))
    print("positive, classe 2: ", positive/len(tokens))