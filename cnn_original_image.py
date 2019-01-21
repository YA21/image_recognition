from models.cnn import cnn_model
from utils.preprocess_images import preprocess_images
from utils.split_data import split_data
from utils.visualize_log import visualize_log
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import os
import pickle
if __name__ == '__main__':
    train_image_dir = "./faces"

    model_output_dir = os.path.join("results",datetime.today().strftime('%Y%m%d_%H%M%S'))
    print("model_output_dir: ",model_output_dir)
    os.makedirs(model_output_dir, exist_ok=True)

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
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_output_dir, "model_{epoch:2d}.h5"),
        monitor='val_loss', period=10)
    history = model.fit(X_train, y_train,
              batch_size=20,
              epochs=100,
              verbose=1,
              validation_data=(X_test, y_test),
              callbacks=[model_checkpoint])
    # visualize training logs
    visualize_log(history, model_output_dir)

    model.save(os.path.join(model_output_dir, "model.h5"))
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("valid loss:", scores[0])
    print("valid accuracy", scores[1])
    print("model summary:")
    model.summary()