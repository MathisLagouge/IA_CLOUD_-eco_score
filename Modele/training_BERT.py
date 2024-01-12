#import function
from creerBERT import *

#param for training
NUM_LABEL = 3
BATCH_SIZE = 4
TEST_SIZE = 0.2
NUM_EPOCH = 5
DATA_PATH = "data/restaurant_stars_int_eq.csv"
SAVE_PATH = "trained_model"

#model training
#save the trained model
#print model accuracy 
train_bert(BATCH_SIZE,NUM_EPOCH,TEST_SIZE,DATA_PATH, SAVE_PATH)
