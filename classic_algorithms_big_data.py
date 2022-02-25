from general_include import *

def decision_trees_classifier(data_x, data_y, verbose=[]):
    print("--- Decision Tree Classifier ---")

    # Split the dataset to train and test
    X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=1)

    # Fit the model
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)

    # Test the model
    Y_pred = model.predict(X_test)
    print('Instances mal classées: %d' % (Y_test != Y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(Y_test, Y_pred))
    print()

    return model

def k_neighbors_classifier(data_x, data_y, n_neighbors=1, verbose=[]):
    print("--- K Neighbors Classifier ---")
    print()

    # data_x = data_x.numpy()
    # data_y = data_y.numpy()
    
    # print("- Data formatting -")
    # data_prov = np.empty( [len(data_x) * len(data_x[0]), len(data_x[0][0])] )
    # iterator = 0
    # for i in range(0, len(data_x)):
    #     for j in range(0, len(data_x[i])):
    #         for k in range(0, len(data_x[i][j])):
    #             data_prov[iterator][k] = data_x[i][j][k]
    #         iterator += 1
    # data_x = np.copy(data_prov)

    # data_prov = np.empty( [len(data_y) * len(data_y[0]), len(data_y[0][0])] )
    # iterator = 0
    # for i in range(0, len(data_y)):
    #     for j in range(0, len(data_y[i])):
    #         for k in range(0, len(data_y[i][j])):
    #             data_prov[iterator][k] = data_y[i][j][k]
    #         iterator += 1
    # data_y = np.copy(data_prov)

    # data_prov = []
    # for i in range(0, len(data_y)):
    #     for j in range(0, len(data_y[i])):
    #         data_prov.append(data_y[i][j])
    # data_y = np.array(data_prov)

    # Split the dataset to train and test
    X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)

    # Fit the model
    print("- Fit the model -")
    print()
    model = KNeighborsClassifier(n_neighbors=n_neighbors, p=1, metric='minkowski')
    model.fit(X_train, Y_train)

    # Test the model
    Y_pred = model.predict(X_test)
    print('Instances mal classées: %d' % (Y_test != Y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(Y_test, Y_pred))
    print('F1: %.2f' % f1_score(Y_test, Y_pred, average='micro'))
    print()

    return model

def random_forest_classifier(data_x, data_y, verbose=[]):
    print("--- Random Forest Classifier ---")

    # Split the dataset to train and test
    X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)

    # Fit the model
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)

    # Test the model
    Y_pred = model.predict(X_test)
    print('Instances mal classées: %d' % (Y_test != Y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(Y_test, Y_pred))
    print('F1: %.2f' % f1_score(Y_test, Y_pred, average='micro'))
    print()

    return model

def evaluate_classic_algorithms(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    print('Instances mal classées: %d' % (Y_test != Y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(Y_test, Y_pred))
    print('F1: %.2f' % f1_score(Y_test, Y_pred, average='micro'))
    print()