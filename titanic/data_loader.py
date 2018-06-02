#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import pandas as pd
import numpy as np
import re


class TitanicData(object):

    def __init__(self, data_dir="data/download"):
        self.train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        self.test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        self._parse_title(self.train_df)
        self._parse_title(self.test_df)
        self._predict_age()
        self._preprocessing()
        self.train_feature = None
        self.train_label = None
        self.train_survived= None
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
            "Embarked": 8,
            "Title": 9,
            "FamilySize": 10
        }
        self.cabin_idx = {}
        self.cabin_count = 0
        self.title_idx = {}
        self.title_count = 0
        self._parse_data()

    def next_train_batch(self, batch_size):
        idx = np.random.choice(len(self.train_feature), batch_size)
        return np.take(self.train_feature, idx, axis=0), np.take(self.train_label, idx, axis=0)

    def _parse_data(self):
        feature = []
        label = []
        survived = []
        for index, row in self.train_df.iterrows():
            feature.append([
                row["PassengerId"],
                row["Pclass"],
                self._transfer_sex(row["Sex"]),
                row["Age"],
                row["SibSp"],
                row["Parch"],
                row["Fare"],
                self._transfer_cabin(row["Cabin"]),
                self._transfer_emparked_port(row["Embarked"]),
                row["Title"],
                1 + row["SibSp"] + row["Parch"]
            ])
            prob = np.zeros(shape=2, dtype=np.float32)
            prob[row["Survived"]] = 1.0
            label.append(prob)
            survived.append(row["Survived"])
        self.train_feature = np.array(feature)
        self.train_label = np.array(label)
        self.train_survived = np.array(survived)
        feature = []
        for index, row in self.test_df.iterrows():
            feature.append([
                row["PassengerId"],
                row["Pclass"],
                self._transfer_sex(row["Sex"]),
                row["Age"],
                row["SibSp"],
                row["Parch"],
                row["Fare"],
                self._transfer_cabin(row["Cabin"]),
                self._transfer_emparked_port(row["Embarked"]),
                row["Title"],
                1 + row["SibSp"] + row["Parch"]
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

    def _transfer_cabin(self, cabin):
        if not pd.isna(cabin):
            cabin = cabin[0]
        if pd.isna(cabin):
            return 0
        elif cabin in self.cabin_idx:
            return self.cabin_idx[cabin]
        else:
            self.cabin_count += 1
            self.cabin_idx[cabin] = self.cabin_count
            return self.cabin_idx[cabin]

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

    @staticmethod
    def _parse_title(df):
        # 正则匹配出名字中间的称谓
        df['Title'] = df['Name'].map(lambda x: re.compile(",(.*?)\.").findall(x)[0])
        # 匹配的逗号后面有空格，记得去除空格，不然下一步没法替换
        df['Title'] = df['Title'].map(str.strip)
        # 替换出5类
        df['Title'][df.Title=='Jonkheer'] = 'Master'
        df['Title'][df.Title.isin(['Ms','Mlle'])] = 'Miss'
        df['Title'][df.Title.isin(['Mme','Dona', 'Lady', 'the Countess'])] = 'Mrs'
        df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Mr'
        df['Title'][df.Title.isin(['Dr','Rev'])] = 'DrAndRev'
        # factorize数值化
        df['Title'] = pd.factorize(df.Title)[0]

    def _predict_age(self):
        from sklearn.ensemble import RandomForestRegressor
        rfc = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
        df = self.train_df[["Title", "Age", "Parch", "Fare", "SibSp", "Pclass"]]
        # df["Sex"] = pd.factorize(df.Sex)[0]
        df['Age'][df.Age.notnull()] = (df['Age'][df.Age.notnull()] / 12).astype(int)
        # df['Age'][(df.Age >= 18) & (df.Age < 55) & (df.Age.notnull())] = 1
        # df['Age'][(df.Age >= 18) & (df.Age.notnull())] = 1
        train_x = df[df.Age.notnull()].drop(['Age'], axis=1).values
        train_y = df['Age'][df.Age.notnull()].values
        rfc.fit(train_x, train_y)
        test_x = df[df.Age.isnull()].drop(['Age'], axis=1).values
        df['Age'][df.Age.isnull()] = rfc.predict(test_x)
        self.train_df['Age'] = df["Age"]
        df = self.test_df[["Title", "Age", "Parch", "Fare", "SibSp", "Pclass"]]
        df["Fare"][df.Fare.isnull()] = 0
        # df["Sex"] = pd.factorize(df.Sex)[0]
        df['Age'][df.Age.notnull()] = (df['Age'][df.Age.notnull()] / 12).astype(int)
        # df['Age'][(df.Age < 18) & (df.Age.notnull())] = 0
        # df['Age'][(df.Age >= 18) & (df.Age < 55) & (df.Age.notnull())] = 1
        # df['Age'][(df.Age >= 18) & (df.Age.notnull())] = 1
        train_x = df[df.Age.notnull()].drop(['Age'], axis=1).values
        train_y = (df['Age'][df.Age.notnull()].values).astype(int)
        print train_x
        tmp = (rfc.predict(train_x)).astype(int)
        print("#########################################")
        print(tmp)
        print(train_y)
        print("age predict accurace: ", np.sum(train_y == tmp) / float(len(tmp)))
        print("#########################################")
        test_x = df[df.Age.isnull()].drop(['Age'], axis=1).values
        df['Age'][df.Age.isnull()] = rfc.predict(test_x)
        self.test_df["Age"] = df["Age"]

    def _preprocessing(self):
        import sklearn.preprocessing as preprocessing
        scaler = preprocessing.StandardScaler()
        self.test_df["Fare"][self.test_df.Fare.isnull()] = 0 # np.mean(self.test_df["Fare"].values)
        # age_scale_param = scaler.fit(self.train_df['Age'].values.reshape(1, -1))
        fare_scale_param = scaler.fit(self.train_df["Fare"].values.reshape(1, -1))
        # self.train_df["Age"] = scaler.fit_transform(self.train_df["Age"].values.reshape(1, -1), age_scale_param).reshape(-1)
        self.train_df["Fare"] = scaler.fit_transform(self.train_df["Fare"].values.reshape(1, -1), fare_scale_param).reshape(-1)
        # self.test_df["Age"] = scaler.fit_transform(self.test_df["Age"].values.reshape(1, -1), age_scale_param).reshape(-1)
        self.test_df["Fare"] = scaler.fit_transform(self.test_df["Fare"].values.reshape(1, -1), fare_scale_param).reshape(-1)
