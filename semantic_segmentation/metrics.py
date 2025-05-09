import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1):
    yt = tf.keras.backend.flatten(y_true)
    yp = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(yt * yp)
    return (2 * intersection + smooth) / (
        tf.reduce_sum(yt) + tf.reduce_sum(yp) + smooth
    )

def iou(y_true, y_pred, smooth=1):
    yt = tf.keras.backend.flatten(y_true)
    yp = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(yt * yp)
    union = tf.reduce_sum(yt) + tf.reduce_sum(yp) - intersection
    return (intersection + smooth) / (union + smooth)
