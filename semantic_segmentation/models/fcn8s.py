from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.models import Model
def build(input_shape=(512,512,3), n_classes=1):
    x_in = Input(shape=input_shape)
    x = Conv2D(64,3,activation="relu",padding="same")(x_in)
    x = Conv2D(64,3,activation="relu",padding="same")(x)
    p1 = MaxPooling2D(2)(x)
    x = Conv2D(128,3,activation="relu",padding="same")(p1)
    x = Conv2D(128,3,activation="relu",padding="same")(x)
    p2 = MaxPooling2D(2)(x)
    x = Conv2D(256,3,activation="relu",padding="same")(p2)
    x = Conv2D(256,3,activation="relu",padding="same")(x)
    p3 = MaxPooling2D(2)(x)
    x = Conv2D(512,3,activation="relu",padding="same")(p3)
    x = Conv2DTranspose(256,4,2,padding="same",activation="relu")(x)
    x = Conv2DTranspose(128,4,2,padding="same",activation="relu")(x)
    x = Conv2DTranspose(64,4,2,padding="same",activation="relu")(x)
    out = Conv2D(n_classes,1,activation="sigmoid")(x)
    return Model(x_in,out,name="FCN8s")
