import os
from glob import glob

import cv2
import matplotlib.pyplot as plt


TEST_DIR = os.path.join(os.getcwd(), "tongue_data", "test_img")
PRED_DIR = os.path.join(os.getcwd(), "prediction")
OUT_DIR = os.path.join(os.getcwd(), "compare")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    test_imgs = sorted(glob(os.path.join(TEST_DIR, "*.*")))

    for p in test_imgs:
        name = os.path.basename(p)
        pred = os.path.join(PRED_DIR, name)
        if not os.path.isfile(pred):
            continue

        img = cv2.cvtColor(cv2.imdecode(np.fromfile(p, dtype=np.uint8), 1), cv2.COLOR_BGR2RGB)
        prd = cv2.cvtColor(cv2.imdecode(np.fromfile(pred, dtype=np.uint8), 1), cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("image")
        plt.axis("off")
        plt.imshow(img)

        plt.subplot(1, 2, 2)
        plt.title("prediction")
        plt.axis("off")
        plt.imshow(prd)

        out = os.path.join(OUT_DIR, name)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()


if __name__ == "__main__":
    main()