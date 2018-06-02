from data_loader import TitanicData 
from model import TitanicModel
import pandas as pd


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def save_result(data, result):
    ids = data.test_feature[:, data.feature_index["PassengerId"]]
    df = pd.DataFrame(data=zip(ids.astype(int), result), columns=["PassengerId", "Survived"])
    df.to_csv("data/result/prediction.csv", index=False)


def train():
    data = TitanicData()
    model = TitanicModel()
    model.train(data)
    result = model.predict(data)
    save_result(data, result)


if __name__ == '__main__':
    train()
