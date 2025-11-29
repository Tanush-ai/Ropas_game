import cv2
import os
import numpy as np
import Hand_Classifier

# Path to the dataset
DATASET_PATH = r"C:\Users\V.Tanush\.cache\kagglehub\datasets\drgfreeman\rockpaperscissors\versions\2"

# Initialize classifier
classifier = Hand_Classifier.HandClassifier()

# Define class mapping
CLASSES = {
    "rock": 1,
    "paper": 2,
    "scissors": 3
}

def train_from_dataset():
    print("Starting training process...")
    
    total_images = 0
    
    for class_name, label in CLASSES.items():
        class_dir = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
            
        print(f"Processing {class_name} images...")
        images = os.listdir(class_dir)
        
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Add to classifier
            classifier.add_sample(img, label)
            total_images += 1
            
            if total_images % 100 == 0:
                print(f"Processed {total_images} images...")

    print(f"Total images processed: {total_images}")
    
    # Train and save
    if classifier.train():
        print("Training complete!")
    else:
        print("Training failed.")

if __name__ == "__main__":
    train_from_dataset()
