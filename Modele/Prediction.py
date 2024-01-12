# On importe les packages
from datasets import load_dataset
from transformers import BertForSequenceClassification, AutoTokenizer
import torch
from datasets import Dataset, DatasetDict

# Donner un score sur les donnees que l'on veut tester

# Tokeniser les donnees que l'on veut tester
def tokenize_data(raw_dataset):

    raw_dataset = Dataset.from_dict(raw_dataset)

    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    def tokenize_function(sentence):
        return tokenizer(sentence['Review'], truncation=True)

    # On tokenise les donnees du dataset
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

    return tokenized_dataset

# On prédit si les donnees sont négatives, neutres ou positives
def prediction(model_path, dataset):

    model = BertForSequenceClassification.from_pretrained(model_path)

    tokens = tokenize_data(dataset)

    # On comptabilise le nombre de reviews négatives, neutres et positives
    nb_review = len(tokens)
    nb_neg = 0
    nb_neu = 0
    nb_pos = 0
    for i in range(len(tokens)):
        input_ids = torch.tensor(tokens[i]['input_ids']).unsqueeze(0)
        attention_mask = torch.tensor(tokens[i]['attention_mask']).unsqueeze(0)

        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        if probabilities[0][0] == max(probabilities[0]):
            nb_neg = nb_neg + 1
        if probabilities[0][1] == max(probabilities[0]):
            nb_neu = nb_neu + 1
        if probabilities[0][2] == max(probabilities[0]):
            nb_pos = nb_pos + 1
        
    return(nb_neg,nb_pos,nb_review)

# Calculer le score obtenu par les donnees predites en fonction du pourcentage de reviews négatives, neutres ou positives
def score(nb_neg,nb_pos,nb_review):

    star = nb_review // 2
    star = star + (nb_pos - nb_neg) // 2
    star = star / nb_review

    # En fonction du pourcentage obtenu, on donne une note
    if star < 0:
        star = 0
    elif star < 0.1:
        star = 0.5
    elif star < 0.2:
        star = 1
    elif star < 0.3:
        star = 1.5
    elif star < 0.4:
        star = 2
    elif star < 0.5:
        star = 2.5
    elif star < 0.6:
        star = 3
    elif star < 0.7:
        star = 3.5
    elif star < 0.8:
        star = 4
    elif star < 0.9:
        star = 4.5
    else:
        star = 5
    
    return star
