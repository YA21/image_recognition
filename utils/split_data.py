from sklearn.model_selection import train_test_split

def split_data(X, Y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    return X_train, X_test, y_train, y_test
