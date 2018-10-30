import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import  RandomForestClassifier



def get_acc(y_pred,y_train,n_sam):
    right_num = (y_train==y_pred).sum()
    print("acc ;" ,right_num/n_sam)

if __name__ == "__main__":
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train_x = train.iloc[:, 1:]
    train_y = train.iloc[:, 0]
    test_x = test
    n_sam = train_x.shape[0]
    pca = PCA(n_components=0.8, whiten=True)  # 0.8
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)

    svc = svm.SVC(kernel='rbf', C=10)
    svc.fit(train_x, train_y)

    y_pre = RF.predict(train_x)
    get_acc(y_pre, train_y,n_sam)
    scores = cross_val_score(svc, train_x, train_y, cv=5)
    print("Accuracy :%0.2f (+- %0.2f)" % (scores.mean(), scores.std() * 2))
    test_y = svc.predict(test_x)
    pd.DataFrame({"ImageId": range(1, len(test_y) + 1), "Label": test_y}).to_csv('out.csv', index=False, header=True)





