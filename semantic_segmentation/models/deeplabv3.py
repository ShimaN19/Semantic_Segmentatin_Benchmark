from tensorflow.keras import applications as ka
def build(input_shape=(512,512,3), n_classes=1):
    return ka.DeepLabV3Plus(weights=None, input_shape=input_shape, classes=n_classes, activation="sigmoid")
