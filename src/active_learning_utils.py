# Uncertainty methods
from model_evaluation_utils import read_testimage, iou, dice_coef, read_testmultimask
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope

def scanPool(selected, method_name, kFold):
    """
    Evaluate uncertainty of samples in the pool using predictions from K-Fold models.

    Parameters:
        selected (list): List to store selected samples with uncertainty scores.
        method_name (str): Method for uncertainty calculation:
                           'shannon_entropy', 'least_confidence', 'margin'.
        kFold (int): Number of models trained using K-Fold cross-validation.

    Returns:
        list: Top 15 samples with the highest uncertainty scores.
    """
    models = []
    # Load models for each fold
    for foldNum in range(kFold):
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
            model_name = f"model{foldNum}.h5"
            models.append(tf.keras.models.load_model(model_name))

    # Iterate over the pool of samples
    for name_x, name_y, name_z in tqdm(zip(pool_x, pool_y, pool_z), total=len(pool_x)):
        name = name_x.split("/")[-1]

        """ Reading the image """
        ori_x, x = read_testimage(name_x)

        # Ensemble predictions from all folds
        for foldNum in range(kFold):
            model = models[foldNum]
            if foldNum == 0:
                y_pred = model.predict(x)
            else:
                y_pred += model.predict(x)
        y_pred = (y_pred / kFold)[0]
        y_pred = y_pred.flatten()

        # Calculate uncertainty based on the specified method
        if method_name == 'shannon_entropy':
            shannon_entropy = -np.sum(y_pred * np.log(y_pred + 1e-9) + (1 - y_pred) * np.log(1 - y_pred + 1e-9))
            selected.append([name_x, name_y, name_z, shannon_entropy])

        elif method_name == 'least_confidence':
            least_confidence = 1 - np.maximum(y_pred, 1 - y_pred)
            least_confidence = np.mean(least_confidence)
            selected.append([name_x, name_y, name_z, least_confidence])

        elif method_name == 'margin':
            margin = -np.abs(y_pred - (1 - y_pred))
            margin = np.mean(margin)
            selected.append([name_x, name_y, name_z, margin])

    # Sort and return the top 15 samples with the highest uncertainty scores
    selected.sort(key=lambda x: x[2], reverse=True)
    return selected[:15]


def saveSampleResult(x, y, z, method, iteration, kFold):
    models = []
    for foldNum in range(kFold):
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
            modelName = "model" + str(foldNum) + ".h5"
            models.append(tf.keras.models.load_model(modelName))
    name = x.split("/")[-1]
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

    filename = str(method) + "_" + str(iteration) + "_" + name
    save_path = f"results2/{filename}"
    save_result(ori_x, ori_y, y_pred, save_path)