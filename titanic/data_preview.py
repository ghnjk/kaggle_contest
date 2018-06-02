from data_loader import TitanicData
import numpy as np
import matplotlib.pyplot as plt 


def show_pclass_distribution(data):
    idx = data.feature_index["Pclass"]
    p_class = data.train_feature[:, idx].reshape(-1)
    is_survived = data.train_label[:].reshape(-1)
    name_list = []
    survied_list = []
    count_list = []
    for c in np.unique(p_class):
        s_count = np.sum(is_survived[p_class == c])
        all_count = np.sum(p_class == c)
        name_list.append("Pclass: %s s_rate: %0.2lf" % (str(int(c)), s_count / float(all_count)))
        survied_list.append(s_count)
        count_list.append(all_count)
    plt.figure()
    plt.bar(range(len(name_list)), survied_list, label="Survived", fc='g', align="center")
    plt.bar(range(len(name_list)), np.array(count_list) - np.array(survied_list), bottom=survied_list, label="all", tick_label=name_list, fc='r', align="center")
    plt.legend()
    plt.title("Survived distribution by Pclass")


def show_sex_distribution(data):
    sex_name = ["female", "male"]
    idx = data.feature_index["Sex"]
    feature = data.train_feature[:, idx].reshape(-1)
    is_survived = data.train_label[:].reshape(-1)
    name_list = []
    survied_list = []
    count_list = []
    for c in np.unique(feature):
        c = int(c)
        s_count = np.sum(is_survived[feature == c])
        all_count = np.sum(feature == c)
        name_list.append("Sex: %s s_rate: %0.2lf" % (sex_name[c], s_count / float(all_count)))
        survied_list.append(s_count)
        count_list.append(all_count)
    plt.figure()
    plt.bar(range(len(name_list)), survied_list, label="Survived", fc='g', align="center")
    plt.bar(range(len(name_list)), np.array(count_list) - np.array(survied_list), bottom=survied_list, label="all", tick_label=name_list, fc='r', align="center")
    plt.legend()
    plt.title("Survived distribution by Sex")


def show_age_distribution(data):
    idx = data.feature_index["Age"]
    feature = (data.train_feature[:, idx].reshape(-1) / 10).astype(int)
    is_survived = data.train_label[:].reshape(-1)
    name_list = []
    survied_list = []
    count_list = []
    for c in np.unique(feature):
        s_count = np.sum(is_survived[feature == c])
        all_count = np.sum(feature == c)
        name_list.append("Age: %s" % (str(c * 10))) 
        survied_list.append(s_count)
        count_list.append(all_count)
    plt.figure()
    plt.bar(range(len(name_list)), survied_list, label="Survived", fc='g', align="center")
    plt.bar(range(len(name_list)), np.array(count_list) - np.array(survied_list), bottom=survied_list, label="all", tick_label=name_list, fc='r', align="center")
    plt.legend()
    plt.title("Survived distribution by Age")


def show_sibsp_distribution(data):
    idx = data.feature_index["SibSp"]
    feature = data.train_feature[:, idx].reshape(-1)
    is_survived = data.train_label[:].reshape(-1)
    name_list = []
    survied_list = []
    count_list = []
    for c in np.unique(feature):
        c = int(c)
        s_count = np.sum(is_survived[feature == c])
        all_count = np.sum(feature == c)
        name_list.append("SibSp: %s" % (str(c)))
        survied_list.append(s_count)
        count_list.append(all_count)
    plt.figure()
    plt.bar(range(len(name_list)), survied_list, label="Survived", fc='g', align="center")
    plt.bar(range(len(name_list)), np.array(count_list) - np.array(survied_list), bottom=survied_list, label="all", tick_label=name_list, fc='r', align="center")
    plt.legend()
    plt.title("Survived distribution by SibSp")


def show_parch_distribution(data):
    idx = data.feature_index["Parch"]
    feature = data.train_feature[:, idx].reshape(-1)
    is_survived = data.train_label[:].reshape(-1)
    name_list = []
    survied_list = []
    count_list = []
    for c in np.unique(feature):
        c = int(c)
        s_count = np.sum(is_survived[feature == c])
        all_count = np.sum(feature == c)
        name_list.append("Parch: %s" % (str(c)))
        survied_list.append(s_count)
        count_list.append(all_count)
    plt.figure()
    plt.bar(range(len(name_list)), survied_list, label="Survived", fc='g', align="center")
    plt.bar(range(len(name_list)), np.array(count_list) - np.array(survied_list), bottom=survied_list, label="all", tick_label=name_list, fc='r', align="center")
    plt.legend()
    plt.title("Survived distribution by Parch")


def show_fare_distribution(data):
    idx = data.feature_index["Fare"]
    feature = (data.train_feature[:, idx].reshape(-1) / 20).astype(int)
    is_survived = data.train_label[:].reshape(-1)
    name_list = []
    survied_list = []
    count_list = []
    for c in np.unique(feature):
        s_count = np.sum(is_survived[feature == c])
        all_count = np.sum(feature == c)
        name_list.append("Fare: %s" % (str(c * 10))) 
        survied_list.append(s_count)
        count_list.append(all_count)
    plt.figure()
    plt.bar(range(len(name_list)), survied_list, label="Survived", fc='g', align="center")
    plt.bar(range(len(name_list)), np.array(count_list) - np.array(survied_list), bottom=survied_list, label="all", tick_label=name_list, fc='r', align="center")
    plt.legend()
    plt.title("Survived distribution by Fare")


def show_embarked_distribution(data):
    idx = data.feature_index["Embarked"]
    feature = data.train_feature[:, idx].reshape(-1)
    is_survived = data.train_label[:].reshape(-1)
    name_list = []
    survied_list = []
    count_list = []
    for c in np.unique(feature):
        c = int(c)
        s_count = np.sum(is_survived[feature == c])
        all_count = np.sum(feature == c)
        name_list.append("Embarked: %s" % (str(c)))
        survied_list.append(s_count)
        count_list.append(all_count)
    plt.figure()
    plt.bar(range(len(name_list)), survied_list, label="Survived", fc='g', align="center")
    plt.bar(range(len(name_list)), np.array(count_list) - np.array(survied_list), bottom=survied_list, label="all", tick_label=name_list, fc='r', align="center")
    plt.legend()
    plt.title("Survived distribution by Embarked")


def preview_data():
    data = TitanicData()
    show_pclass_distribution(data)
    show_sex_distribution(data)
    show_age_distribution(data)
    show_sibsp_distribution(data)
    show_parch_distribution(data)
    show_fare_distribution(data)
    show_embarked_distribution(data)
    plt.show()


if __name__ == '__main__':
    preview_data()

