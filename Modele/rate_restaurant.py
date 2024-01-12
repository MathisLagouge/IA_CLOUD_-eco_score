# import les packages
import pandas as pd
from Prediction import prediction, score
from datasets import load_dataset

# Categories que l'on note
CATEGORIES = ["organic", "climate", "water", "social", "governance", "waste", "adverse"]
# Dataframe final
DF = {}
# Chemin du modèle que l'on utilise
MODEL_PATH = "trained_model"

# On cree un fichier csv pour la ville, regroupant toutes les notes des restaurants recuperes par categories et la note globale
def creation_csv(city):
    # Pour chaque categorie
    for i in range(len(CATEGORIES)):
        print("Traitement de la categorie ", CATEGORIES[i])
        # On charge le csv associe
        datas = load_dataset("csv", data_files=str("Scrapping/categories/" + city + "/" + CATEGORIES[i] + ".csv"))["train"]
        liste_restaurant = []
        iter = 0
        # On recupere tous les restaurants et leurs reviews
        while(iter < len(datas["PLACE"])):
            if datas[iter]["PLACE"] not in liste_restaurant:
                avis = []
                liste_restaurant.append(datas[iter]["PLACE"])
                for j in range(len(datas["PLACE"])):
                    if(datas[j]["PLACE"] == liste_restaurant[-1]):
                        avis.append(datas[j]["REVIEW"])
                df = {"Review" : avis}
                pred = prediction(MODEL_PATH, df)
                res = score(pred[0], pred[1], pred[2])
                if(liste_restaurant[-1] not in DF.keys()):
                    DF[liste_restaurant[-1]] = {"Review_"+CATEGORIES[i] : res}
                else:
                    DF[liste_restaurant[-1]]["Review_"+CATEGORIES[i]] = res
            iter += 1

    # On cree un dictionnaire contenant sur chaque ligne, le nom du restaurant et les notes associees
    final = {"PLACE" : [],
            "organic" : [],
            "climate" : [],
            "water" : [],
            "social" : [],
            "governance" : [],
            "waste" : [],
            "adverse" : [],
            "mean" : []}
    for x in DF.keys():
        final["PLACE"].append(x)
        nb_categories = 0
        mean = 0
        for y in CATEGORIES:
            try:
                final[y].append(DF[x]["Review_"+y])
                nb_categories += 1
                mean += DF[x]["Review_"+y]
            except:
                final[y].append(None)
        final["mean"].append(mean / nb_categories)

    # On enregistre le dataframe dans un csv
    dataframe = pd.DataFrame(final)
    dataframe.to_csv("Scrapping/categories/" + city + "/notation_restaurant.csv", index=False)
