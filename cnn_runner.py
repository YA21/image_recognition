from models.cnn import cnn_model
from utils.preprocess_images import preprocess_images

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess_images(images_dir="./images")
    model = cnn_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
    model.fit(X_train, y_train,
              batch_size=128,
              epochs=20,
              verbose=1,
              validation_data=(X_test, y_test))
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("valid loss:", scores[0])
    print("valid accuracy", scores[1])
