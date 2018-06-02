import os
import pandas as pd
import numpy as np


class TitanicData(object):

    def __init__(self, data_dir="data/download"):
        self.train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        self.test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        self.train_feature = None
        self.train_label = None
        self.test_feature = None
        self.feature_index = {
            "PassengerId": 0,
            "Pclass": 1,
            "Sex": 2,
            "Age": 3,
            "SibSp": 4,
            "Parch": 5,
            "Fare": 6,
            "Cabin": 7,
            "Embarked": 8
        }
        self.carbin_idx = {}
        self.carbin_count = 0
        self._parse_data()

    def next_train_batch(self, batch_size):
        idx = np.random.choice(len(self.train_feature), batch_size)
        return np.take(self.train_feature, idx, axis=0), np.take(self.train_label, idx, axis=0)

    def _parse_data(self):
        feature = []
        label = []
        for index, row in self.train_df.iterrows():
            feature.append([
                row["PassengerId"],
                row["Pclass"],
                self._transfer_sex(row["Sex"]),
                self._normal_age(row["Age"]),
                row["SibSp"],
                row["Parch"],
                row["Fare"],
                self._transfer_carbin(row["Cabin"]),
                self._transfer_emparked_port(row["Embarked"])
            ])
            prob = np.zeros(shape=2, dtype=np.float32)
            prob[row["Survived"]] = 1.0
            label.append(prob)
        self.train_feature = np.array(feature)
        self.train_label = np.array(label)
        feature = []
        for index, row in self.test_df.iterrows():
            feature.append([
                row["PassengerId"],
                row["Pclass"],
                self._transfer_sex(row["Sex"]),
                self._normal_age(row["Age"]),
                row["SibSp"],
                row["Parch"],
                row["Fare"],
                self._transfer_carbin(row["Cabin"]),
                self._transfer_emparked_port(row["Embarked"])
            ])
        self.test_feature = np.array(feature)
       
    @staticmethod
    def _transfer_sex(sex):
        if sex == "male":
            return 1
        elif sex == "female":
            return 0
        else:
            raise Exception("invalid sex vaue: " + sex)

    @staticmethod
    def _normal_age(age):
        if pd.isna(age):
            return 0
        else:
            return int(age)

    def _transfer_carbin(self, carbin):
        if carbin in self.carbin_idx:
            return self.carbin_idx[carbin]
        elif pd.isna(carbin):
            return 0
        else:
            self.carbin_count += 1
            self.carbin_idx[carbin] = self.carbin_count
            return self.carbin_idx[carbin]

    @staticmethod
    def _transfer_emparked_port(port):
        if pd.isna(port):
            return 0
        elif port == "C":
            return 1
        elif port == 'Q':
            return 2
        elif port == 'S':
            return 3
        else:
            raise Exception("invalid port: " + str(port))
