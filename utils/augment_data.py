from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt

def augment_data(original_data, augment_num=5):
      datagen = ImageDataGenerator(
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

      original_data = original_data.reshape((1,) + original_data.shape)

      augmented_data = []

      i = 0
      #plt.figure(figsize=(20,8))
      for index, batch in enumerate(datagen.flow(original_data, batch_size=1)):
            augmented_data.append(batch[0])

            # plt.imshow(image.array_to_img(batch[0]))
            # plt.savefig("sample.png")

            if index == augment_num:
                  break

      return augmented_data