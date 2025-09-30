import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd 
import matplotlib.pyplot as plt

def load_intel_dataset(
    root_dir="IntelImageClassification",
    img_size=(224, 224),
    val_size=0.1,
    random_seed=42,
    verbose=1,
    selected_classes=None   # 👈 new argument
):
    """
    Load the Intel Image Classification dataset from Kaggle.

    Handles both:
        Intel/seg_train/buildings/...
        Intel/seg_test/buildings/...
    and the case with nested folders:
        Intel/seg_train/seg_train/buildings/...
        Intel/seg_test/seg_test/buildings/...

    Args:
        root_dir: Path to the extracted Intel dataset.
        img_size: Tuple for resizing (H, W).
        val_size: Fraction of training set reserved for validation.
        random_seed: For reproducibility.
        verbose: Print info if True.
        selected_classes: list of class names to include (default: all)

    Returns:
        dict with numpy arrays:
            'X_train', 'y_train'
            'X_val',   'y_val'
            'X_test',  'y_test'
            'class_names'
    """

    def resolve_subdir(base_dir):
        """Handle double-nested case (seg_train/seg_train)."""
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if len(subdirs) == 1 and subdirs[0].lower().startswith("seg_"):
            return os.path.join(base_dir, subdirs[0])
        return base_dir

    def load_images_from_dir(base_dir, img_size, split_name=""):
        class_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

        # If selected_classes is set → filter
        if selected_classes is not None:
            class_names = [cls for cls in class_names if cls in selected_classes]

        class_to_idx = {cls: i for i, cls in enumerate(class_names)}

        image_paths, labels = [], []
        for cls in class_names:
            cls_dir = os.path.join(base_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(cls_dir, fname))
                    labels.append(class_to_idx[cls])

        total = len(image_paths)
        arr = []
        for i, p in enumerate(image_paths, start=1):
            with Image.open(p) as img:
                img = img.convert("RGB").resize(img_size, Image.LANCZOS)
                arr.append(np.asarray(img, dtype=np.float32) / 255.0)

            # Print progress every 500 images (or last one)
            if verbose and (i % 500 == 0 or i == total):
                print(f"[{split_name}] Loaded {i}/{total} images")

        X = np.stack(arr, axis=0) if arr else np.empty((0, *img_size, 3), dtype=np.float32)
        y = np.array(labels)
        return X, y, class_names

    # Fix nested dirs if needed
    train_dir = resolve_subdir(os.path.join(root_dir, "seg_train"))
    test_dir  = resolve_subdir(os.path.join(root_dir, "seg_test"))

    # Load train and test sets with simple progress logging
    X_train_all, y_train_all, class_names = load_images_from_dir(train_dir, img_size, split_name="train")
    X_test, y_test, _ = load_images_from_dir(test_dir, img_size, split_name="test")

    # Split validation from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all,
        test_size=val_size,
        stratify=y_train_all,
        random_state=random_seed
    )

    if verbose:
        print(f"Final shapes → Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print("Classes:", class_names)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "class_names": class_names
    }

   



def plot_training_history(training_history_object, list_of_metrics=None):
    #Plots training and validation curves from the keras History object.
    history_dict = training_history_object.history
    if list_of_metrics is None:
        list_of_metrics = [
            key for key in list(history_dict.keys()) if 'val_' not in key
        ]
    trainHistDF = pd.DataFrame(history_dict)
    train_keys = list_of_metrics
    valid_keys = ['val_' + key for key in train_keys]
    nr_plots = len(train_keys)
    fig, ax = plt.subplots(1, nr_plots, figsize=(5*nr_plots, 4))
    for i in range(len(train_keys)):
        ax[i].plot(np.array(trainHistDF[train_keys[i]]), label='Training')
        ax[i].plot(np.array(trainHistDF[valid_keys[i]]), label='Validation')
        ax[i].set_xlabel('Epoch')
        ax[i].set_title(train_keys[i])
        ax[i].grid('on')
        ax[i].legend()
    fig.tight_layout()
    plt.show()


# -------------------------
# Custom F1-score metric
# -------------------------
def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.argmax(y_pred, axis=1), tf.float32)
    y_true = tf.cast(tf.argmax(y_true, axis=1), tf.float32)

    tp = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred) & tf.equal(y_true, 1), tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.equal(y_true, 0) & tf.equal(y_pred, 1), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.equal(y_true, 1) & tf.equal(y_pred, 0), tf.float32))

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return f1


# -------------------------
# Helper: Resize dataset
# -------------------------
def resize_dataset(X, target_size=(32, 32)):
    if X is None or X.size == 0:
        return X
    return tf.image.resize(X, target_size).numpy().astype("float32")    
    




# -------------------------
# Corruptions
# -------------------------
def add_gaussian_noise(images, mean=0.0, std=0.1):
    noise = np.random.normal(mean, std, images.shape)
    noisy = images + noise
    return np.clip(noisy, 0., 1.)

def add_motion_blur(images, ksize=9):
    # simple 1D blur kernel
    kernel = np.zeros(ksize)
    kernel[:] = 1.0 / ksize
    
    blurred = []
    for img in images:
        # apply horizontal blur
        img_h = convolve1d(img, kernel, axis=1, mode='reflect')
        # apply vertical blur
        img_hv = convolve1d(img_h, kernel, axis=0, mode='reflect')
        blurred.append(img_hv)
    return np.array(blurred)    