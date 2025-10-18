#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coronary Artery Disease Classifier (Normal vs Abnormal)
- TensorFlow/Keras (CNN via EfficientNetB0 transfer learning)
- Robust preprocessing, augmentation, class weighting
- Long training with EarlyStopping + ModelCheckpoint
- Full evaluation: confusion matrix + classification report
- Designed to achieve >90% val/test accuracy on typical two-class datasets

USAGE:
    python cad_cnn_training_tf.py --data_root "/mnt/data/cad_dataset_extracted" --epochs 50 --img_size 224 --batch 32

Requires: tensorflow>=2.10, scikit-learn, matplotlib
"""

import os, argparse, math, json, itertools, datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.metrics import classification_report, confusion_matrix

def build_datasets(data_root, img_size=224, batch_size=32, val_split=0.2, test_split=0.1, seed=42):
    data_root = Path(data_root)
    # Assume structure: .../Dataset/normal, .../Dataset/abnormal
    # If an extra nesting exists, search for the folder that contains the class subfolders
    candidate = data_root
    # auto-detect folder containing class dirs
    for p in data_root.rglob('*'):
        if p.is_dir():
            subs = [d.name.lower() for d in p.iterdir() if d.is_dir()]
            if 'normal' in subs and 'abnormal' in subs:
                candidate = p
                break

    class_names = sorted([d.name for d in candidate.iterdir() if d.is_dir()])
    print("Using dataset directory:", candidate)
    print("Classes:", class_names)

    # First, create a temp split: train+val and test
    ds_all = tf.keras.preprocessing.image_dataset_from_directory(
        candidate,
        labels='inferred',
        label_mode='int',
        image_size=(img_size, img_size),
        shuffle=True,
        seed=seed,
        batch_size=batch_size,
        validation_split=val_split + test_split,
        subset='training'
    )
    ds_temp = tf.keras.preprocessing.image_dataset_from_directory(
        candidate,
        labels='inferred',
        label_mode='int',
        image_size=(img_size, img_size),
        shuffle=True,
        seed=seed,
        batch_size=batch_size,
        validation_split=val_split + test_split,
        subset='validation'
    )

    # Now split ds_temp into val and test
    val_batches = int(len(ds_temp) * (val_split / (val_split + test_split)))
    ds_val = ds_temp.take(val_batches)
    ds_test = ds_temp.skip(val_batches)

    AUTOTUNE = tf.data.AUTOTUNE

    # Data augmentation
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
    ], name="augmentation")

    def prep(x, y):
        x = tf.cast(x, tf.float32)
        x = preprocess_input(x)  # EfficientNet scale
        return x, y

    ds_train = ds_all.map(lambda x,y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.map(prep, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    ds_val = ds_val.map(prep, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    ds_test = ds_test.map(prep, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    # Count items for class weights
    y_all = []
    for _, yb in tf.keras.preprocessing.image_dataset_from_directory(
        candidate, labels='inferred', label_mode='int',
        image_size=(img_size, img_size), batch_size= batch_size, shuffle=False
    ):
        y_all.extend(yb.numpy().tolist())

    y_all = np.array(y_all)
    class_counts = np.bincount(y_all)
    total = y_all.shape[0]
    class_weights = {i: float(total)/(len(class_counts)*c) for i, c in enumerate(class_counts)}
    print("Class counts:", class_counts.tolist())
    print("Class weights:", class_weights)

    return ds_train, ds_val, ds_test, class_weights, class_names

def build_model(img_size=224, num_classes=2, train_backbone=False, dropout=0.3):
    inputs = keras.Input((img_size, img_size, 3))
    base = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    base.trainable = train_backbone  # start frozen
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--fine_tune_at', type=int, default=200, help='Unfreeze last N layers for fine-tuning')
    args = ap.parse_args()

    ds_train, ds_val, ds_test, class_weights, class_names = build_datasets(
        args.data_root, img_size=args.img_size, batch_size=args.batch
    )

    model = build_model(img_size=args.img_size, num_classes=len(class_names), train_backbone=False, dropout=0.3)
    model.compile(optimizer=keras.optimizers.Adam(args.lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    ckpt_dir = Path('checkpoints')
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    ckpt_path = str(ckpt_dir / 'best.keras')

    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_accuracy', mode='max'),
        keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True, monitor='val_accuracy', mode='max'),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, verbose=1, monitor='val_loss')
    ]

    print("Stage 1: Train head only (backbone frozen)")
    hist1 = model.fit(ds_train, validation_data=ds_val, epochs=max(10, args.epochs//3),
                      class_weight=class_weights, callbacks=callbacks)

    # Fine-tune
    print("Stage 2: Fine-tune last layers")
    base = model.layers[1]  # EfficientNet backbone
    for layer in base.layers[-args.fine_tune_at:]:
        layer.trainable = True

    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    hist2 = model.fit(ds_train, validation_data=ds_val, epochs=args.epochs,
                      class_weight=class_weights, callbacks=callbacks)

    # Evaluate
    print("Evaluating on validation and test...")
    val_metrics = model.evaluate(ds_val, verbose=0)
    test_metrics = model.evaluate(ds_test, verbose=0)
    print("Val metrics [loss, acc]:", val_metrics)
    print("Test metrics [loss, acc]:", test_metrics)

    # Classification report
    y_true, y_pred = [], []
    for xb, yb in ds_test:
        pb = model.predict(xb, verbose=0)
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(np.argmax(pb, axis=1).tolist())

    print("\nClassification report (TEST):")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Save model
    model.save('cad_efficientnet_b0.keras')
    print("Saved model to cad_efficientnet_b0.keras")

if __name__ == '__main__':
    main()
