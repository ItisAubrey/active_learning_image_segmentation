
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from tensorflow.keras.utils import CustomObjectScope
# Global constants

H = 256
W = 256
def read_testimage(path):
    """
    Reads and processes a test image.

    Parameters:
        path (str): Path to the image file.

    Returns:
        ori_x (numpy.ndarray): Original image (H, W, 3).
        x (numpy.ndarray): Normalized and expanded image (1, H, W, 3).
    """
    # Read the image in color
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    # Resize the image to the desired dimensions
    x = cv2.resize(x, (W, H))
    ori_x = x.copy()  # Preserve the original image

    # Normalize the pixel values to the range [0, 1]
    x = x / 255.0
    x = x.astype(np.float32)

    # Add a batch dimension
    x = np.expand_dims(x, axis=0)  # Shape: (1, 256, 256, 3)

    return ori_x, x
def read_testmultimask(path1, path2):
    """
    Reads and processes multi-channel masks.

    Parameters:
        path1 (str): Path to the first mask file.
        path2 (str): Path to the second mask file.

    Returns:
        ori_x (numpy.ndarray): Original stacked masks (H, W, 2).
        x (numpy.ndarray): Normalized, resized, and thresholded masks (H, W, 2).
    """
    # Read the masks in grayscale
    x1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    x2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

    # Stack the two masks along the last axis
    ori_x = np.stack((x1, x2), axis=-1)  # Shape: (H, W, 2)

    # Resize and normalize the masks
    x1 = cv2.resize(x1, (W, H)) / 255.0
    x2 = cv2.resize(x2, (W, H)) / 255.0

    # Stack the processed masks
    x = np.stack((x1, x2), axis=-1)  # Shape: (256, 256, 2)

    # Apply thresholding to binarize the masks
    x = (x > 0.5).astype(np.int32)

    return ori_x, x
def read_testsinglemask(path):
    """
    Reads and processes a single-channel mask.

    Parameters:
        path (str): Path to the mask file.

    Returns:
        ori_x (numpy.ndarray): Original mask (H, W).
        x (numpy.ndarray): Normalized, resized, and thresholded mask (H, W).
    """
    # Read the mask in grayscale
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Resize the mask to the desired dimensions
    x = cv2.resize(x, (W, H))
    ori_x = x.copy()  # Preserve the original mask

    # Normalize and threshold the mask
    x = x / 255.0
    x = (x > 0.5).astype(np.int32)

    return ori_x, x

def save_result(ori_x, ori_y, y_pred, save_path):
    """
    Combines and saves the original image, ground truth mask, and predicted mask.

    Parameters:
        ori_x (numpy.ndarray): Original image (H, W, 3).
        ori_y (numpy.ndarray): Ground truth mask (H, W, 2).
        y_pred (numpy.ndarray): Predicted mask (H, W, 2).
        save_path (str): Path to save the concatenated result.

    Returns:
        None
    """
    # Create a white separator line
    line = np.ones((H, 10, 3)) * 255  # Shape: (256, 10, 3)

    # Prepare the ground truth mask as a 3-channel image
    ori_y_3d = np.zeros((H, W, 3))  # Empty RGB mask
    ori_y_3d[:, :, 0:2] = ori_y  # Assign the mask to red and green channels
    ori_y_3d = 255 - ori_y_3d  # Invert the colors for better contrast

    # Prepare the predicted mask as a 3-channel image
    y_pred_3d = np.zeros((H, W, 3))  # Empty RGB mask
    y_pred_3d[:, :, 0:2] = y_pred  # Assign the mask to red and green channels
    y_pred_3d = (1 - y_pred_3d) * 255.0  # Invert and scale to [0, 255]

    # Concatenate the original image, ground truth, and prediction
    cat_images = np.concatenate([ori_x, line, ori_y_3d, line, y_pred_3d], axis=1)

    # Save the combined image
    cv2.imwrite(save_path, cat_images)

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        y_truef = tf.keras.layers.Flatten()(y_true)
        y_predf = tf.keras.layers.Flatten()(y_pred)
        if np.sum(y_predf)==0 | np.sum(y_truef)==0:
            y_true=1-y_true
            y_pred=1-y_pred
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


def dice_coef(y_true, y_pred):
    #smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    if np.sum(y_pred)==0 | np.sum(y_true)==0:
        y_true=1-y_true
        y_pred=1-y_pred
    intersection = tf.reduce_sum(y_true * y_pred)
    outcome=(2* intersection) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))
    #print(outcome)
    return outcome

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
def evaluation(SCORE, kFold):
    models = []
    for foldNum in range(kFold):
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
            modelName = "model" + str(foldNum) + ".h5"
            models.append(tf.keras.models.load_model(modelName))
    for x, y, z in tqdm(zip(test_x, test_y, test_z), total=len(test_x)):
        name = x.split("/")[-1]

        """ Reading the image and mask """
        ori_x, x = read_testimage(x)
        ori_y, y = read_testmultimask(y, z)

        for foldNum in range(kFold):
            model = models[foldNum]
            if foldNum == 0:
                y_pred = model.predict(x)
            else:
                y_pred += model.predict(x)
        y_pred = (y_pred / foldNum)[0] > 0.5
        # y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred.astype(np.int32)
        # print(y_pred.shape)
        # save_path = f"results/{name}"
        # save_result(ori_x, ori_y, y_pred, save_path)

        """ Flattening the numpy arrays. """
        y = y.flatten()
        y_pred = y_pred.flatten()
        # print(f"y length: {len(y)}, y_pred length: {len(y_pred)}")
        """ Calculating metrics values """
        # acc_value = accuracy_score(y, y_pred)
        iou_value = iou(y, y_pred)
        dic_coef_value = dice_coef(y, y_pred)
        dice_loss_value = 1 - dic_coef_value
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
        SCORE.append([name, iou_value, dic_coef_value, dice_loss_value, f1_value])


def evaluationTest(SCORE, kFold):
    models = []
    for foldNum in range(kFold):
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
            modelName = "model" + str(foldNum) + ".h5"
            models.append(tf.keras.models.load_model(modelName))
    for xn, yn, zn in tqdm(zip(test_x, test_y, test_z), total=len(test_x)):
        name = xn.split("/")[-1]

        """ Reading the image and mask """
        ori_x, x = read_testimage(xn)
        ori_y, y = read_testsinglemask(yn)
        ori_z, z = read_testsinglemask(zn)

        for foldNum in range(kFold):
            model = models[foldNum]
            if foldNum == 0:
                y_pred = model.predict(x)
            else:
                y_pred += model.predict(x)
        y_pred = (y_pred / foldNum)[0] > 0.5
        # y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred.astype(np.int32)
        print("y pred shape: ", y_pred.shape)
        # print(y_pred.shape)
        # save_path = f"results/{name}"
        # save_result(ori_x, ori_y, y_pred, save_path)
        y_pred1 = y_pred[:, :, 0]
        y_pred2 = y_pred[:, :, 1]
        """ Flattening the numpy arrays. """
        y = y.flatten()
        z = z.flatten()
        y_pred1 = y_pred1.flatten()
        y_pred2 = y_pred2.flatten()
        print(f"y length: {len(y)}, z length: {len(z)}, y_pred1 length: {len(y_pred1)} ,y_pred2 length: {len(y_pred2)}")
        # acc_value = accuracy_score(y, y_pred)
        iou_value1 = iou(y, y_pred1)
        iou_value2 = iou(z, y_pred2)
        dic_coef_value1 = dice_coef(y, y_pred1)
        dic_coef_value2 = dice_coef(z, y_pred2)
        dice_loss_value1 = 1 - dic_coef_value1
        dice_loss_value2 = 1 - dic_coef_value2
        f1_value1 = f1_score(y, y_pred1, labels=[0, 1], average="binary")
        f1_value2 = f1_score(z, y_pred2, labels=[0, 1], average="binary")
        # jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
        # recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
        # precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
        SCORE.append([name, iou_value1, iou_value2, dic_coef_value1, dic_coef_value2,
                      dice_loss_value1, dice_loss_value2, f1_value1, f1_value2])
        saveSampleResult(xn, yn, zn, 9, 19, 3)