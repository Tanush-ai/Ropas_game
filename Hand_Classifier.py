import cv2
import numpy as np
import os

class HandClassifier:
    def __init__(self, model_path="model.xml"):
        self.samples = []
        self.labels = []
        self.model = cv2.ml.KNearest_create()
        self.is_trained = False
        self.img_size = (32, 32) # Smaller = faster processing
        self.model_path = model_path
        self.load_model()

    def process_image(self, img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize
        resized = cv2.resize(gray, self.img_size)
        # Flatten
        return resized.reshape(-1).astype(np.float32)

    def add_sample(self, img, label):
        # label: 1=Rock, 2=Paper, 3=Scissor
        features = self.process_image(img)
        self.samples.append(features)
        self.labels.append(label)
        # print(f"Added sample for class {label}. Total samples: {len(self.samples)}")

    def train(self):
        if len(self.samples) < 3:
            print("Not enough samples to train.")
            return False
        
        samples_array = np.array(self.samples, dtype=np.float32)
        labels_array = np.array(self.labels, dtype=np.int32)
        
        self.model.train(samples_array, cv2.ml.ROW_SAMPLE, labels_array)
        self.is_trained = True
        print("Model trained successfully!")
        self.save_model()
        return True

    def save_model(self):
        try:
            self.model.save(self.model_path)
            print(f"Model saved to {self.model_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = cv2.ml.KNearest_load(self.model_path)
                self.is_trained = True
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")

    def predict(self, img):
        if not self.is_trained:
            return 0 # Unknown
            
        features = self.process_image(img)
        features = features.reshape(1, -1)
        
        ret, results, neighbours, dist = self.model.findNearest(features, k=3)
        return int(results[0][0])
