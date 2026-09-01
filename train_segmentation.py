from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# =========================================================
# PATHS
# =========================================================

IMAGE_DIR = Path("segmentation/images/images")
MASK_DIR = Path("segmentation/masks/masks")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 10
SEED = 42


# =========================================================
# CHECK FOLDERS
# =========================================================

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        "segmentation/images folder not found."
    )

if not MASK_DIR.exists():
    raise FileNotFoundError(
        "segmentation/masks folder not found."
    )


# =========================================================
# GET IMAGE FILES
# =========================================================

image_files = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png"))
)

mask_files = sorted(
    list(MASK_DIR.glob("*.jpg")) +
    list(MASK_DIR.glob("*.jpeg")) +
    list(MASK_DIR.glob("*.png"))
)

print("Images found:", len(image_files))
print("Masks found:", len(mask_files))


if len(image_files) == 0:
    raise ValueError(
        "No images found inside segmentation/images."
    )

if len(mask_files) == 0:
    raise ValueError(
        "No masks found inside segmentation/masks."
    )


# =========================================================
# CREATE PAIRS USING FILE NAME
# =========================================================

mask_dictionary = {
    mask.stem: mask
    for mask in mask_files
}

pairs = []

for image in image_files:

    if image.stem in mask_dictionary:

        pairs.append(
            (
                str(image),
                str(mask_dictionary[image.stem])
            )
        )


print("Matched image-mask pairs:", len(pairs))


if len(pairs) < 10:
    raise ValueError(
        "Not enough matching image-mask pairs found. "
        "Check that image and mask filenames match."
    )


# =========================================================
# SHUFFLE
# =========================================================

rng = np.random.default_rng(SEED)

rng.shuffle(pairs)


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

split_index = int(
    len(pairs) * 0.8
)

train_pairs = pairs[:split_index]
val_pairs = pairs[split_index:]

print("Training pairs:", len(train_pairs))
print("Validation pairs:", len(val_pairs))


# =========================================================
# LOAD IMAGE + MASK
# =========================================================

def load_image_mask(image_path, mask_path):

    # Image
    image = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0


    # Mask
    mask = tf.io.read_file(mask_path)

    mask = tf.image.decode_image(
        mask,
        channels=1,
        expand_animations=False
    )

    mask = tf.image.resize(
        mask,
        IMG_SIZE,
        method="nearest"
    )

    mask = tf.cast(
        mask,
        tf.float32
    ) / 255.0

    # Convert mask to binary
    mask = tf.where(
        mask > 0.5,
        1.0,
        0.0
    )

    return image, mask


# =========================================================
# DATASET
# =========================================================

def create_dataset(pairs):

    image_paths = [
        pair[0]
        for pair in pairs
    ]

    mask_paths = [
        pair[1]
        for pair in pairs
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            mask_paths
        )
    )

    dataset = dataset.map(
        load_image_mask,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_ds = create_dataset(
    train_pairs
)

val_ds = create_dataset(
    val_pairs
)


# =========================================================
# U-NET MODEL
# =========================================================

def conv_block(x, filters):

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    return x


def build_unet():

    inputs = layers.Input(
        shape=IMG_SIZE + (3,)
    )

    # Encoder
    c1 = conv_block(
        inputs,
        32
    )

    p1 = layers.MaxPooling2D()(c1)


    c2 = conv_block(
        p1,
        64
    )

    p2 = layers.MaxPooling2D()(c2)


    c3 = conv_block(
        p2,
        128
    )

    p3 = layers.MaxPooling2D()(c3)


    # Bottleneck
    c4 = conv_block(
        p3,
        256
    )


    # Decoder
    u5 = layers.UpSampling2D()(c4)

    u5 = layers.Concatenate()(
        [u5, c3]
    )

    c5 = conv_block(
        u5,
        128
    )


    u6 = layers.UpSampling2D()(c5)

    u6 = layers.Concatenate()(
        [u6, c2]
    )

    c6 = conv_block(
        u6,
        64
    )


    u7 = layers.UpSampling2D()(c6)

    u7 = layers.Concatenate()(
        [u7, c1]
    )

    c7 = conv_block(
        u7,
        32
    )


    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c7)

    return models.Model(
        inputs,
        outputs
    )


model = build_unet()


# =========================================================
# COMPILE
# =========================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# =========================================================
# TRAIN
# =========================================================

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


# =========================================================
# SAVE MODEL
# =========================================================

model_path = (
    MODEL_DIR /
    "tumor_segmentation.keras"
)

model.save(
    model_path
)

print()
print("======================================")
print("Segmentation training completed!")
print("Model saved to:")
print(model_path)
print("======================================")