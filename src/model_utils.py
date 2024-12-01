from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    UpSampling2D,
    Concatenate,
    Input,
    Activation
)
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.utils import CustomObjectScope

from data_processing_utils import tf_dataset
def conv_block(x, num_filters):
    """Convolutional block with two convolution layers."""
    x = Conv2D(num_filters, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(num_filters, (3, 3), activation="relu", padding="same")(x)
    return x

def build_model(input_shape=(256, 256, 3)):
    """Builds a standard U-Net model."""
    inputs = Input(input_shape)

    ## Encoder
    c1 = conv_block(inputs, 64)
    p1 = MaxPool2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = MaxPool2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = MaxPool2D((2, 2))(c3)

    c4 = conv_block(p3, 512)
    p4 = MaxPool2D((2, 2))(c4)

    ## Bridge
    c5 = conv_block(p4, 1024)

    ## Decoder
    u6 = UpSampling2D((2, 2))(c5)
    concat6 = Concatenate()([u6, c4])
    c6 = conv_block(concat6, 512)

    u7 = UpSampling2D((2, 2))(c6)
    concat7 = Concatenate()([u7, c3])
    c7 = conv_block(concat7, 256)

    u8 = UpSampling2D((2, 2))(c7)
    concat8 = Concatenate()([u8, c2])
    c8 = conv_block(concat8, 128)

    u9 = UpSampling2D((2, 2))(c8)
    concat9 = Concatenate()([u9, c1])
    c9 = conv_block(concat9, 64)

    ## Output Layer
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

    return Model(inputs, outputs)


# train the model using augmented data
def trainModel(train_x, train_y, train_z, valid_x, valid_y, valid_z, folder,
               batch_size=5, lr=0.001, num_epochs=80):
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    # Hyperparaqmeters
    # batch_size = 5
    # lr = 0.001   ## 0.0001
    # num_epochs = 80

    model_path = "model" + str(folder) + ".h5"
    # csv_path = "data1.csv"

    """Load Augmented Dataset For Training """
    # new_path = "new_data/"
    print(f"start new folder, Train: {len(train_x)} - {len(train_y)} - {len(train_z)}")

    train_dataset = tf_dataset(train_x, train_y, train_z, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, valid_z, batch=batch_size)

    train_steps = (len(train_x) // batch_size)
    valid_steps = (len(valid_x) // batch_size)
    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    """ Model """
    model = build_model()
    metrics = [dice_coef, iou, Recall(), Precision()]
    # metrics = [Recall(), Precision()]
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr), metrics=metrics)
    # else:
    # with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
    # model = tf.keras.models.load_model("fullmodel.h5")
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        # CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )

    return model
def crossVali(n_split,epoch):
    X=np.array(train_x)
    Y=np.array(train_y)
    Z=np.array(train_z)
    folderNum=0
    for train_index,valid_index in KFold(n_split).split(X):
        x_train,x_valid=X[train_index],X[valid_index]
        y_train,y_valid=Y[train_index],Y[valid_index]
        z_train,z_valid=Z[train_index],Z[valid_index]
        model=trainModel(train_x=x_train.tolist(), train_y=y_train.tolist(), train_z=z_train.tolist(),
                         valid_x=x_valid.tolist(), valid_y=y_valid.tolist(), valid_z=z_valid.tolist(),folder=folderNum,
               batch_size = 5, lr = 0.001, num_epochs = epoch)
        folderNum += 1