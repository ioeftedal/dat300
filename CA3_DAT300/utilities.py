import os, glob, random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras import backend as K

# -------------------------
# Constants for mask lookup
# -------------------------
MASK_SUFFIXES = ["_mask"]   # allowed suffixes for mask filenames
MASK_EXTS = [".png"]        # allowed mask file extensions

# -------------------------
# Helper: Find mask for image
# -------------------------
def find_mask_for(img_path, msk_dir):
    """Find the corresponding mask file for a given image path."""
    base = os.path.splitext(os.path.basename(img_path))[0]
    search_dirs = [msk_dir, os.path.dirname(img_path)]
    for d in search_dirs:
        for ext in MASK_EXTS:
            cand = os.path.join(d, base + ext)
            if os.path.exists(cand):
                return cand
        for suf in MASK_SUFFIXES:
            for ext in MASK_EXTS:
                cand = os.path.join(d, base + suf + ext)
                if os.path.exists(cand):
                    return cand
    raise FileNotFoundError(f"Mask not found for {img_path}")

# -------------------------
# Helper: Build (image, mask) pairs
# -------------------------
def build_pairs(img_dir, msk_dir):
    """Return (image, mask) pairs and a list of missing images."""
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    pairs, missing = [], []
    for p in img_paths:
        try:
            pairs.append((p, find_mask_for(p, msk_dir)))
        except FileNotFoundError:
            missing.append(p)
    return pairs, missing



# -------------------------
# Helper: Report dataset statistics
# -------------------------
def report_pairs(pairs, missing, max_show=5):
    """Print dataset statistics and optionally show missing examples."""
    print(f"Total images: {len(pairs) + len(missing)}")
    print(f"Matched pairs: {len(pairs)}")
    print(f"Missing masks: {len(missing)}")
    if missing[:max_show]:
        print("Examples missing:", [os.path.basename(x) for x in missing[:max_show]])


# -------------------------
# Helper: Peek at dataset
# -------------------------
def peek_pairs(pairs, n_show=3):
    """Visualize a few random image–mask pairs."""
    plt.figure(figsize=(12, 4*n_show))
    for i, (img_path, msk_path) in enumerate(random.sample(pairs, n_show)):
        img = plt.imread(img_path)
        msk = plt.imread(msk_path)
        plt.subplot(n_show, 2, 2*i+1)
        plt.imshow(img, cmap="gray"); plt.axis("off"); plt.title(f"Image {os.path.basename(img_path)}")
        plt.subplot(n_show, 2, 2*i+2)
        plt.imshow(msk, cmap="gray"); plt.axis("off"); plt.title("Mask")
    plt.tight_layout()
    plt.show()


# -------------------------
# Helper: Split dataset
# -------------------------
def split_dataset(pairs, seed=42, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train/val/test sets given ratios."""
    paths_train, paths_tmp = train_test_split(pairs, test_size=val_ratio+test_ratio, random_state=seed)
    rel_test = test_ratio / (val_ratio+test_ratio)
    paths_val, paths_test = train_test_split(paths_tmp, test_size=rel_test, random_state=seed)
    return paths_train, paths_val, paths_test


# -------------------------
# Helper: Plot split distribution
# -------------------------
def plot_split_distribution(paths_train, paths_val, paths_test):
    """Plot dataset split distribution as a pie chart."""
    sizes = [len(paths_train), len(paths_val), len(paths_test)]
    labels = ["Train", "Val", "Test"]
    plt.figure(figsize=(5,5))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=["skyblue","orange","lime"])
    plt.title("Dataset Split")
    plt.show()

# -------------------------
# Helper: Losses and Metrics
# -------------------------    
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

bce = tf.keras.losses.BinaryCrossentropy()

def bce_dice_loss(y_true, y_pred):
    return 0.5 * bce(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

def iou_soft(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred - y_true*y_pred, axis=[1,2,3])
    iou = (inter + smooth) / (union + smooth)
    return tf.reduce_mean(iou)
    


def f1_score(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    P = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (P + K.epsilon())

    Pred_P = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = TP / (Pred_P + K.epsilon())
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# -------------------------
# Config
# -------------------------
IMG_SIZE = (512, 512)     # Change if you run out of memory
BATCH_SIZE = 4

# -------------------------
# Helper: Read image
# -------------------------
def read_image(path):
    """Read and preprocess an image into float32 [0,1] range, resized to IMG_SIZE."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)  # force 3ch
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    return img

# -------------------------
# Helper: Read mask
# -------------------------
def read_mask(path):
    """Read and preprocess a mask into {0,1}, resized to IMG_SIZE."""
    m = tf.io.read_file(path)
    m = tf.image.decode_image(m, channels=1, expand_animations=False)  # uint8
    m = tf.image.resize(m, IMG_SIZE, method="nearest")                 # nearest keeps labels
    m = tf.cast(m, tf.float32) / 255.0                                # normalize to [0,1]
    m = tf.where(m > 0.5, 1.0, 0.0)                                   # binarize
    return m


# -------------------------
# Helper: Load (image, mask) pair
# -------------------------
def load_pair(img_path, msk_path):
    """Return an (image, mask) tensor pair."""
    img = read_image(img_path)
    msk = read_mask(msk_path)
    return img, msk

# -------------------------
# Helper: Augmentation
# -------------------------
def augment(img, msk):
    """Apply mask-safe random augmentations to (image, mask)."""
    # Horizontal flip
    flip = tf.less(tf.random.uniform([]), 0.5)
    img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
    msk = tf.cond(flip, lambda: tf.image.flip_left_right(msk), lambda: msk)

    # Light random zoom/shift using pad + random_crop
    PAD = 16
    img = tf.image.resize_with_pad(img, IMG_SIZE[0] + PAD, IMG_SIZE[1] + PAD)
    msk = tf.image.resize_with_pad(msk, IMG_SIZE[0] + PAD, IMG_SIZE[1] + PAD)

    stacked = tf.concat([img, msk], axis=-1)  # HxWx4
    stacked = tf.image.random_crop(stacked, size=[IMG_SIZE[0], IMG_SIZE[1], 4])
    img, msk = stacked[..., :3], stacked[..., 3:]

    # Light intensity jitter (image only)
    img = tf.image.random_brightness(img, max_delta=0.05)
    img = tf.image.random_contrast(img, 0.95, 1.05)
    return img, msk

# -------------------------
# Helper: Preprocess for ResNet
# -------------------------
def preprocess(img, msk):
    """Prepare image for ResNet (0..255 scaling + mean subtraction)."""
    img = resnet_preprocess(img * 255.0)
    return img, msk

# -------------------------
# Helper: Make tf.data.Dataset
# -------------------------
def make_dataset(pairs, augment_data=False, batch_size=BATCH_SIZE, shuffle=True):
    """Build tf.data.Dataset from (image, mask) pairs."""
    img_paths = [p[0] for p in pairs]
    msk_paths = [p[1] for p in pairs]

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
    dataset = dataset.map(lambda i, m: load_pair(i, m), num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment_data:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(pairs))
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def plot_training_history(training_history_object, list_of_metrics=None):
    history_dict = training_history_object.history
    keys = history_dict.keys()

    if list_of_metrics is None:
        # pick all non-val metrics
        list_of_metrics = [k for k in keys if not k.startswith("val_") and k != "loss"]

    train_keys = list_of_metrics
    valid_keys = ['val_' + key for key in train_keys]

    nr_plots = len(train_keys)
    fig, ax = plt.subplots(1, nr_plots, figsize=(5*nr_plots, 4))
    for i, key in enumerate(train_keys):
        ax[i].plot(history_dict[key], label="Training")
        ax[i].plot(history_dict[valid_keys[i]], label="Validation")
        ax[i].set_xlabel("Epoch")
        ax[i].set_title(key)
        ax[i].grid(True)
        ax[i].legend()
    plt.tight_layout()
    plt.show()


def plot_training_history_and_return(training_history_object, list_of_metrics=['Accuracy', 'F1_score', 'IoU']):
    """
    Description: This is meant to be used in scripts run on Orion
    Input:
        training_history_object:: training history object returned from 
                                  tf.keras.model.fit()
        list_of_metrics        :: Can be any combination of the following options 
                                  ('Loss', 'Precision', 'Recall' 'F1_score', 'IoU'). 
                                  Generates one subplot per metric, where training 
                                  and validation metric is plotted.
    Output:
    """
    rawDF = pd.DataFrame(training_history_object.history)
    plotDF = pd.DataFrame()

    plotDF['Accuracy']     = (rawDF['true_positives'] + rawDF['true_negatives']) / (rawDF['true_positives'] + rawDF['true_negatives'] + rawDF['false_positives'] + rawDF['false_negatives'])
    plotDF['val_Accuracy'] = (rawDF['val_true_positives'] + rawDF['val_true_negatives']) / (rawDF['val_true_positives'] + rawDF['val_true_negatives'] + rawDF['val_false_positives'] + rawDF['val_false_negatives'])

    plotDF['IoU']          = rawDF['true_positives'] / (rawDF['true_positives'] + rawDF['false_positives'] + rawDF['false_negatives'])
    plotDF['val_IoU']      = rawDF['val_true_positives'] / (rawDF['val_true_positives'] + rawDF['val_false_positives'] + rawDF['val_false_negatives'])
    
    plotDF['F1_score']     = rawDF['F1_score']
    plotDF['val_F1_score'] = rawDF['val_F1_score']

    train_keys = list_of_metrics
    valid_keys = ['val_' + key for key in list_of_metrics]
    nr_plots = len(list_of_metrics)
    fig, ax = plt.subplots(1,nr_plots,figsize=(5*nr_plots,4))
    for i in range(len(list_of_metrics)):
        ax[i].plot(np.array(plotDF[train_keys[i]]), label='Training')
        ax[i].plot(np.array(plotDF[valid_keys[i]]), label='Validation')
        ax[i].set_xlabel('Epoch')
        ax[i].set_title(list_of_metrics[i])
        ax[i].grid('on')
        ax[i].legend()
    fig.tight_layout
    return fig


def visualize_predictions(model, dataset, n=3):
    """Show n samples with image, ground truth mask, and predicted mask."""
    for imgs, msks in dataset.take(1):  # take 1 batch
        preds = model.predict(imgs)
        preds = (preds > 0.5).astype(np.uint8)  # threshold at 0.5
        
        for i in range(n):
            plt.figure(figsize=(12,4))
            
            # Original image
            plt.subplot(1,3,1)
            plt.imshow((imgs[i].numpy() + 1) / 2)  # undo preprocessing if needed
            plt.axis("off")
            plt.title("Original Image")
            
            # Ground truth mask
            plt.subplot(1,3,2)
            plt.imshow(msks[i].numpy().squeeze(), cmap="gray")
            plt.axis("off")
            plt.title("Ground Truth Mask")
            
            # Predicted mask
            plt.subplot(1,3,3)
            plt.imshow(preds[i].squeeze(), cmap="gray")
            plt.axis("off")
            plt.title("Predicted Mask")
            
            plt.show()

    