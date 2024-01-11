import pandas as pd
from Prediction import prediction, score
from datasets import load_dataset

CATEGORIES = ["organic", "climate", "water", "social", "governance", "waste", "adverse"]
DF = {}
MODEL_PATH = "Restaurant_Stars"
CITY = "New York"

def creation_csv(city):
    for i in range(len(CATEGORIES)):
        print("Traitement de la categorie ", CATEGORIES[i])
        datas = load_dataset("csv", data_files=str(city + "/" + CATEGORIES[i] + ".csv"))["train"]
        liste_restaurant = []
        iter = 0
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

    final = {"PLACE" : [],
            "organic" : [],
            "climate" : [],
            "water" : [],
            "social" : [],
            "governance" : [],
            "waste" : [],
            "adverse" : []}
    for x in DF.keys():
        final["PLACE"].append(x)
        for y in CATEGORIES:
            try:
                final[y].append(DF[x]["Review_"+y])
            except:
                final[y].append(None)

    dataframe = pd.DataFrame(final)
    dataframe.to_csv('notation_restaurant_' + city + '.csv', index=False)

creation_csv(CITY)