from models.cnn import cnn_model
from utils.preprocess_images import preprocess_images
from utils.split_data import split_data

if __name__ == '__main__':
    train_image_dir = "./images"
    X, Y, class_index = preprocess_images(images_dir=train_image_dir)
    X_train, X_test, y_train, y_test = split_data(X, Y, test_size=0.2)

    # if you're doing binary classification, only one output layer is enough
    if len(class_index)==2:
        output_layer_num = 1
        y_train = y_train[:,0]
        y_test = y_test[:,0]
    else:
        output_layer_num = y_train.shape[1]

    model = cnn_model(input_shape=X_train.shape[1:], output_layer_num=output_layer_num)
    model.fit(X_train, y_train,
              batch_size=128,
              epochs=2,
              verbose=1,
              validation_data=(X_test, y_test))

    scores = model.evaluate(X_test, y_test, verbose=1)
    print("valid loss:", scores[0])
    print("valid accuracy", scores[1])
    print("model summary:")
    model.summary()
