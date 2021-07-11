import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

def generate_data_1(directory, augmentation, batchsize, file_list, label_1):
    i = 0
    while True:
        image_batch = []
        label_1_batch = []
        for b in range(batchsize):
            if i == (len(file_list)):
                i = 0
            img = image.load_img(directory + '/' + file_list.iloc[i] + '.jpg', grayscale=False, target_size=(384, 384))
            img = image.img_to_array(img)

            if augmentation:
                datagen = ImageDataGenerator(
                    rotation_range=360,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.2,
                    zoom_range=0.2,
                    channel_shift_range=20,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode="nearest")
                img = datagen.random_transform(img)
                img = img / 255.0
            else:
                img = img / 255.0

            image_batch.append(img)
            label_1_batch.append(label_1.iloc[i])
            i = i + 1

        yield (np.asarray(image_batch), np.asarray(label_1_batch))

def generate_data_2(directory, augmentation, batch_size, file_list, label_1, label_2, sample_weights):

    i = 0
    image_batch_r = []
    label_1_batch_r = []
    label_2_batch_r = []
    sample_weight_r = []
    while True:
        image_batch = []
        label_1_batch = []
        label_2_batch = []
        sample_weight = []
        for b in range(batch_size):
            if i == (len(file_list)):
                i = 0

            img = image.load_img(directory + '/' + file_list.iloc[i] + '.jpg', grayscale=False, target_size=(384, 384))
            img = image.img_to_array(img)
            if augmentation:
                datagen = ImageDataGenerator(
                    rotation_range=360,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.2,
                    zoom_range=0.2,
                    channel_shift_range=20,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode="nearest")
                img = datagen.random_transform(img)
                img = img / 255.0
            else:
                img = img / 255.0

            image_batch.append(img)
            label_1_batch.append(label_1.iloc[i])
            label_2_batch.append(label_2[i])
            sample_weight.append(sample_weights[i])

            #print(file_list.iloc[i] + ',' + str(label_1.iloc[i]) + ',' + str(label_2[i]) + ',' + str(sample_weights[i]))
            i = i + 1

        if all(sample == 0 for sample in sample_weight):
            print('all weights zero')
            yield (
                np.asarray(image_batch_r),
                ({'out_class': np.asarray(label_1_batch_r), 'out_asymm': np.asarray(label_2_batch_r)}),
                ({'out_asymm': np.asarray(sample_weight_r)}))
        else:
            image_batch_r = image_batch  # Memory, when all the samples for asymmetry score is zero generator returns previous batch instead of zeros
            label_1_batch_r = label_1_batch
            label_2_batch_r = label_2_batch
            sample_weight_r = sample_weight
            yield (
                np.asarray(image_batch),
                {'out_class': np.asarray(label_1_batch), 'out_asymm': np.asarray(label_2_batch)},
                {'out_asymm': np.asarray(sample_weight)})