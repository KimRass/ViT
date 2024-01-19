import cv2
from PIL import Image
import numpy as np
from time import time
from datetime import timedelta
from pathlib import Path


def print_number_of_parameters(model):
    print(f"""{sum([p.numel() for p in model.parameters()]):,}""")


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))


def load_image(img_path):
    img_path = str(img_path)
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    return img


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def show_image(img):
    copied = img.copy()
    copied = _to_pil(copied)
    copied.show()


def _apply_jet_colormap(img):
    img_jet = cv2.applyColorMap(src=(255 - img), colormap=cv2.COLORMAP_JET)
    return img_jet


def _to_array(img):
    img = np.array(img)
    return img


def _blend_two_images(img1, img2, alpha=0.5):
    img1 = _to_pil(img1)
    img2 = _to_pil(img2)
    img_blended = Image.blend(im1=img1, im2=img2, alpha=alpha)
    return _to_array(img_blended)


def _to_3d(img):
    if img.ndim == 2:
        return np.dstack([img, img, img])
    else:
        return img


def _preprocess_image(img):
    if img.dtype == "bool":
        img = img.astype("uint8") * 255
        
    if img.ndim == 2:
        if (
            np.array_equal(np.unique(img), np.array([0, 255])) or
            np.array_equal(np.unique(img), np.array([0])) or
            np.array_equal(np.unique(img), np.array([255]))
        ):
            img = _to_3d(img)
        else:
            img = _apply_jet_colormap(img)
    return img


def _blend_two_images(img1, img2, alpha=0.5):
    img1 = _to_pil(img1)
    img2 = _to_pil(img2)
    img_blended = Image.blend(im1=img1, im2=img2, alpha=alpha)
    return _to_array(img_blended)


def save_image(img1, img2=None, alpha=0.5, path="") -> None:
    copied1 = _preprocess_image(
        _to_array(img1.copy())
    )
    if img2 is None:
        img_arr = copied1
    else:
        copied2 = _to_array(
            _preprocess_image(
                _to_array(img2.copy())
            )
        )
        img_arr = _to_array(
            _blend_two_images(img1=copied1, img2=copied2, alpha=alpha)
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if img_arr.ndim == 3:
        cv2.imwrite(
            filename=str(path), img=img_arr[:, :, :: -1], params=[cv2.IMWRITE_JPEG_QUALITY, 100]
        )
    elif img_arr.ndim == 2:
        cv2.imwrite(
            filename=str(path), img=img_arr, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
        )
