import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from albumentations.augmentations.geometric.transforms import RandomRotate90, GridDistortion
from albumentations.augmentations.transforms import HorizontalFlip, VerticalFlip


# === Image and Mask Reading ===
def read_image(path):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256)) / 255.0
    return image


def read_multiMask(path1, path2):
    path1, path2 = path1.decode(), path2.decode()
    mask1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    mask1, mask2 = cv2.resize(mask1, (256, 256)) / 255.0, cv2.resize(mask2, (256, 256)) / 255.0
    return np.stack((mask1, mask2), axis=-1)


def check_empty_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256)) / 255.0
    return np.sum(mask.flatten()) == 0


# === TensorFlow Dataset Parsing ===
def tf_parse(x, y, z):
    def _parse(x, y, z):
        x = read_image(x)
        y = read_multiMask(y, z)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y, z], [tf.float64, tf.float64])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 2])
    return x, y


def tf_dataset(x, y, z, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y, z))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch).repeat()
    return dataset


# === Utility Functions ===
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# === Data Augmentation ===
#Data augmentation ,each image 5 transformations


def augment_data(images, masks1, masks2, save_path, augRatio=0.5):
    H, W = 256, 256

    for x, y, z in tqdm(zip(images, masks1, masks2), total=len(images)):
        """ Extracting the name and extension of the image and the masks """
        image_name, image_extn = os.path.splitext(os.path.basename(x))
        mask1_name, mask1_extn = os.path.splitext(os.path.basename(y))
        mask2_name, mask2_extn = os.path.splitext(os.path.basename(z))

        """ Reading and enhancing the image """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        r_image, g_image, b_image = cv2.split(x)
        r_image_eq = cv2.equalizeHist(r_image)
        g_image_eq = cv2.equalizeHist(g_image)
        b_image_eq = cv2.equalizeHist(b_image)
        x = cv2.merge((r_image_eq, g_image_eq, b_image_eq))

        """ Adjusting masks to 3 channels """
        y = cv2.imread(y, cv2.IMREAD_COLOR)
        y[:, :, 0] = y[:, :, 2]
        y[:, :, 1] = y[:, :, 2]

        z = cv2.imread(z, cv2.IMREAD_COLOR)
        z[:, :, 2] = z[:, :, 0]
        z[:, :, 1] = z[:, :, 0]

        """ Applying augmentations """
        save_images = [x]
        save_masks1 = [y]
        save_masks2 = [z]

        # Define augmentations
        augmentations = [
            RandomRotate90(p=1.0),
            GridDistortion(p=1.0),
            HorizontalFlip(p=1.0),
            VerticalFlip(p=1.0),
        ]

        for aug in augmentations:
            augmented = aug(image=x, mask=y)
            aug_image, aug_mask1 = augmented['image'], augmented['mask']

            augmented = aug(image=x, mask=z)
            _, aug_mask2 = augmented['image'], augmented['mask']

            save_images.append(aug_image)
            save_masks1.append(aug_mask1)
            save_masks2.append(aug_mask2)

        """ Saving the original and augmented images and masks """
        for idx, (i, m1, m2) in enumerate(zip(save_images, save_masks1, save_masks2)):
            i = cv2.resize(i, (W, H))
            m1 = cv2.resize(m1, (W, H))
            m2 = cv2.resize(m2, (W, H))

            tmp_img_name = f"{image_name}_{idx}{image_extn}"
            tmp_mask1_name = f"{mask1_name}_{idx}{mask1_extn}"
            tmp_mask2_name = f"{mask2_name}_{idx}{mask2_extn}"

            image_path = os.path.join(save_path, "images", tmp_img_name)
            mask1_path = os.path.join(save_path, "masks1", tmp_mask1_name)
            mask2_path = os.path.join(save_path, "masks2", tmp_mask2_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask1_path, m1)
            cv2.imwrite(mask2_path, m2)
            # Increment the index
            idx += 1

def load_data(path, split2=0.2, split3=0.2, augRatio=0.5):
    images = sorted(glob(os.path.join(path, "tiff/*")))
    masks1 = sorted(glob(os.path.join(path, "lesion/*")))
    masks2 = sorted(glob(os.path.join(path, "solid/*")))
    print("original image num:", len(images))

    # Creating folders.
    create_dir("new_data_multi/images")
    create_dir("new_data_multi/masks1")
    create_dir("new_data_multi/masks2")

    # Applying data augmentationon training dataset
    augment_data(images, masks1, masks2, "new_data_multi", augRatio)
    path2 = "new_data_multi/"
    images = sorted(glob(os.path.join(path2, "images/*")))
    masks1 = sorted(glob(os.path.join(path2, "masks1/*")))
    masks2 = sorted(glob(os.path.join(path2, "masks2/*")))
    print("after augmentation, image num:", len(images))
    # size1 = int(len(images) * split1)
    size2 = int(len(images) * split2)
    size3 = int(len(images) * split3)

    # train_x, valid_x = train_test_split(images, test_size=size1, random_state=42)
    # train_y, valid_y = train_test_split(masks, test_size=size1, random_state=42)

    train_x, test_x = train_test_split(images, test_size=size2, random_state=42)
    train_y, test_y = train_test_split(masks1, test_size=size2, random_state=42)
    train_z, test_z = train_test_split(masks2, test_size=size2, random_state=42)

    pool_x, train_x = train_test_split(train_x, test_size=size3, random_state=42)
    pool_y, train_y = train_test_split(train_y, test_size=size3, random_state=42)
    pool_z, train_z = train_test_split(train_z, test_size=size3, random_state=42)

    return (train_x, train_y, train_z), (pool_x, pool_y, pool_z), (test_x, test_y, test_z)

def saveList(path,saveList):
    with open(path, "w") as f:
        for s in saveList:
            f.write(str(s) +"\n")

def openList(path):
    openedList=[]
    with open(path, "r") as f:
        for line in f:
            openedList.append(line.strip())
    return openedList