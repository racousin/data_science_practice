"""
Dataset Preparation Script for Boat Detection Exercise
This script creates train/test datasets for boat object detection using YOLO format
"""

import os
import shutil
import pickle
from pathlib import Path
from PIL import Image
import numpy as np

# Define paths
DATASET_ROOT = Path("/Users/raphaelcousin/data_science_practice/website/public/modules/data-science-practice/module7/course/ship_yolo_dataset")
OUTPUT_DIR = Path("/Users/raphaelcousin/data_science_practice/website/public/modules/data-science-practice/module7/exercise")

def load_image_and_labels(split='train'):
    """Load images and their corresponding YOLO labels"""
    images_dir = DATASET_ROOT / split / 'images'
    labels_dir = DATASET_ROOT / split / 'labels'

    images = []
    labels = []
    filenames = []

    # Get all image files
    image_files = sorted(list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')))

    for img_path in image_files:
        # Load image
        img = Image.open(img_path)
        img_array = np.array(img)

        # Load corresponding label
        label_path = labels_dir / f"{img_path.stem}.txt"

        if label_path.exists():
            # Read YOLO format labels
            boxes = []
            with open(label_path, 'r') as f:
                for line in f:
                    # YOLO format: class_id x_center y_center width height (normalized)
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        boxes.append([class_id, x_center, y_center, width, height])

            if boxes:  # Only include images with at least one box
                images.append(img_array)
                labels.append(np.array(boxes))
                filenames.append(img_path.name)

    return images, labels, filenames

def create_datasets():
    """Create training and test datasets"""
    print("Loading training data...")
    X_train, y_train, train_files = load_image_and_labels('train')

    print(f"Loaded {len(X_train)} training images")
    print(f"Sample label shape: {y_train[0].shape if y_train else 'No labels'}")

    print("\nLoading validation/test data...")
    X_test, y_test_target, test_files = load_image_and_labels('val')

    print(f"Loaded {len(X_test)} test images")
    print(f"Sample test label shape: {y_test_target[0].shape if y_test_target else 'No labels'}")

    # Save datasets as pickle files
    print("\nSaving datasets...")

    # Save training data
    with open(OUTPUT_DIR / 'X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    print(f"✓ Saved X_train.pkl ({len(X_train)} images)")

    with open(OUTPUT_DIR / 'y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    print(f"✓ Saved y_train.pkl ({len(y_train)} label arrays)")

    with open(OUTPUT_DIR / 'train_files.pkl', 'wb') as f:
        pickle.dump(train_files, f)
    print(f"✓ Saved train_files.pkl")

    # Save test data
    with open(OUTPUT_DIR / 'X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    print(f"✓ Saved X_test.pkl ({len(X_test)} images)")

    with open(OUTPUT_DIR / 'y_test_target.pkl', 'wb') as f:
        pickle.dump(y_test_target, f)
    print(f"✓ Saved y_test_target.pkl ({len(y_test_target)} label arrays)")

    with open(OUTPUT_DIR / 'test_files.pkl', 'wb') as f:
        pickle.dump(test_files, f)
    print(f"✓ Saved test_files.pkl")

    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Count total boxes
    train_boxes = sum(len(boxes) for boxes in y_train)
    test_boxes = sum(len(boxes) for boxes in y_test_target)

    print(f"Training boxes: {train_boxes}")
    print(f"Test boxes: {test_boxes}")
    print(f"Avg boxes per train image: {train_boxes/len(y_train):.2f}")
    print(f"Avg boxes per test image: {test_boxes/len(y_test_target):.2f}")

    # Sample image info
    if X_train:
        print(f"\nSample image shape: {X_train[0].shape}")
        print(f"Sample boxes:\n{y_train[0]}")

if __name__ == "__main__":
    create_datasets()
    print("\n✅ Dataset preparation complete!")
