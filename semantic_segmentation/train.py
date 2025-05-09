import argparse, pathlib, datetime, tensorflow as tf
from segbench import get_model
from segbench.metrics import dice_coef, iou

def _ds(root, img_size=(512, 512)):
    imgs = sorted((root / "images").glob("*"))
    masks = sorted((root / "masks").glob("*"))

    def load(i, m):
        x = tf.image.resize(
            tf.image.decode_png(tf.io.read_file(str(i)), channels=3), img_size
        ) / 255.0
        y = tf.image.resize(
            tf.image.decode_png(tf.io.read_file(str(m)), channels=1),
            img_size,
            method="nearest",
        ) / 255.0
        return x, y

    ds = (
        tf.data.Dataset.from_tensor_slices((imgs, masks))
        .map(load, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(4)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=pathlib.Path)
    ap.add_argument("--model", default="hybrid_u")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", type=pathlib.Path, default="runs")
    args = ap.parse_args()

    ds = _ds(args.dataset)
    val = ds.take(1)
    train = ds.skip(1)

    model = get_model(args.model)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[dice_coef, iou],
    )

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ck = args.out / f"{args.model}_{ts}.h5"
    ck.parent.mkdir(parents=True, exist_ok=True)

    cb = tf.keras.callbacks.ModelCheckpoint(
        ck, monitor="val_iou", save_best_only=True, mode="max"
    )
    model.fit(train, validation_data=val, epochs=args.epochs, callbacks=[cb])
    print(f"Best weights saved to {ck}")

if __name__ == "__main__":
    main()
