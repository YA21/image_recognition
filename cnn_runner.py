from models.cnn import cnn_model
from utils.preprocess_images import preprocess_images
from utils.split_data import split_data

if __name__ == '__main__':
    image_dir = "./images"
    X, Y = preprocess_images(images_dir=image_dir)
    X_train, X_test, y_train, y_test = split_data(X, Y, test_size=0.2)
    model = cnn_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
    model.fit(X_train, y_train,
              batch_size=128,
              epochs=2,
              verbose=1,
              validation_data=(X_test, y_test))
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("valid loss:", scores[0])
    print("valid accuracy", scores[1])
