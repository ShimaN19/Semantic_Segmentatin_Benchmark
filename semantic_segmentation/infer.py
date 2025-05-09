import argparse, pathlib, cv2, numpy as np
from segbench import get_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=pathlib.Path)
    ap.add_argument("--model", default="hybrid_u")
    ap.add_argument("--weights", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    model = get_model(args.model)
    model.load_weights(args.weights)

    img = cv2.cvtColor(cv2.imread(str(args.image)), cv2.COLOR_BGR2RGB)
    img_r = cv2.resize(img, model.input_shape[1:3]) / 255.0

    pred = model.predict(np.expand_dims(img_r, 0))[0, ..., 0]
    mask = (pred > 0.5).astype(np.uint8) * 255

    out = args.out or args.image.with_name(args.image.stem + "_mask.png")
    cv2.imwrite(str(out), mask)
    print("Saved", out)

if __name__ == "__main__":
    main()
