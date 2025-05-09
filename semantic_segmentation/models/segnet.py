from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
def build(input_shape=(512,512,3), n_classes=1):
    inp = Input(shape=input_shape)
    e1 = Conv2D(64,3,activation="relu",padding="same")(inp); p1 = MaxPooling2D(2)(e1)
    e2 = Conv2D(128,3,activation="relu",padding="same")(p1); p2 = MaxPooling2D(2)(e2)
    b = Conv2D(256,3,activation="relu",padding="same")(p2)
    d2 = Conv2D(128,3,activation="relu",padding="same")(UpSampling2D(2)(b))
    d1 = Conv2D(64,3,activation="relu",padding="same")(UpSampling2D(2)(d2))
    out = Conv2D(n_classes,1,activation="sigmoid")(d1)
    return Model(inp,out,name="SegNet")
