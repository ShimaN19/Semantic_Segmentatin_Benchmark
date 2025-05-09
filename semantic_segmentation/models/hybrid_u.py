from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose, MaxPooling2D,
                                     BatchNormalization, Activation, Dropout, concatenate)
from tensorflow.keras.models import Model

DEFAULT_FILTERS = (16, 32, 64, 128, 256)
DEFAULT_DROPOUT = 0.10

def _conv_block(x, f, k=3, bn=True):
    x = Conv2D(f, k, padding="same", kernel_initializer="he_normal")(x)
    if bn:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(f, k, padding="same", kernel_initializer="he_normal")(x)
    if bn:
        x = BatchNormalization()(x)
    return Activation("relu")(x)

def build(input_shape=(512,512,3), n_classes=1, n_filters=DEFAULT_FILTERS, dropout=DEFAULT_DROPOUT, bn=True):
    inp = Input(shape=input_shape)
    c1 = _conv_block(inp, n_filters[0], bn=bn); p1 = Dropout(dropout)(MaxPooling2D(2)(c1))
    c2 = _conv_block(p1, n_filters[1], bn=bn); p2 = Dropout(dropout)(MaxPooling2D(2)(c2))
    c3 = _conv_block(p2, n_filters[2], bn=bn); p3 = Dropout(dropout)(MaxPooling2D(2)(c3))
    c4 = _conv_block(p3, n_filters[3], bn=bn); p4 = Dropout(dropout)(MaxPooling2D(2)(c4))
    c5 = _conv_block(p4, n_filters[4], bn=bn)
    u6 = concatenate([Conv2DTranspose(n_filters[3],3,2,padding="same")(c5), c4])
    c6 = _conv_block(Dropout(dropout)(u6), n_filters[3], bn=bn)
    u7 = concatenate([Conv2DTranspose(n_filters[2],3,2,padding="same")(c6), c3])
    c7 = _conv_block(Dropout(dropout)(u7), n_filters[2], bn=bn)
    u8 = concatenate([Conv2DTranspose(n_filters[1],3,2,padding="same")(c7), c2])
    c8 = _conv_block(Dropout(dropout)(u8), n_filters[1], bn=bn)
    u9 = concatenate([Conv2DTranspose(n_filters[0],3,2,padding="same")(c8), c1])
    c9 = _conv_block(Dropout(dropout)(u9), n_filters[0], bn=bn)
    out = Conv2D(n_classes,1,activation="sigmoid",name="mask")(c9)
    return Model(inp, out, name="HybridU")
