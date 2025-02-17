import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

class ImageClassifier:
    def _init_(self, image_folders, n_estimators=100, random_state=60):
        self.image_folders = image_folders
        self.n_estimators = n_estimators
        self.random_state = random_state
        
    def load_images(self):
        """
        Load images and labels from specified folders.
        """
        imgs, labels = [], []
        for folder in self.image_folders:
            path = os.path.join("path_to_dataset", folder)  # Adjust path as needed
            for file in os.listdir(path):
                img_path = os.path.join(path, file)
                if img_path.lower().endswith((".jpg", ".jpeg")):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (128, 128))
                    img_data = np.array(img).flatten()
                    imgs.append(img_data)
                    labels.append(folder)
        self.imgs = np.array(imgs)
        self.labels = np.array(labels)
        print("Images of given dataset loaded ... ")
        print("."*20)
        print("."*20)
        
    def train_test_split(self, test_size=0.2):
        """
        Split the dataset into training and testing sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.imgs, self.labels, test_size=test_size, random_state=self.random_state
        )
        print("Data has been split...")
        print("."*20)
        print("."*20)

    def train_model(self):
        """
        Train the Random Forest classifier.
        """
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)
        print("Initializing Training Of Model ..")
        print("."*20)
        print("."*20)

    def evaluate(self):
        """
        Evaluate the performance of the trained model.
        """
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average='weighted')
        print("Evaluation of  the performance of the trained model...")
        print("."*20)
        print("."*20)

        print("Actual\tPredicted")
        for i in range(len(self.y_test)):
            print("."*13)
            print(f"  {os.path.basename(self.y_test[i])}\t{os.path.basename(predictions[i])}")
        print("-"*20)
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        print("-"*20)
        
    def predict(self, new_image_path):
        """
        Predict the label of a new image.
        """
        try:
            new_img = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)
            new_img = cv2.resize(new_img, (128, 128))
            new_img_data = np.array(new_img).flatten()
            predicted_label = self.model.predict([new_img_data])[0]
            print("Predicted label:", os.path.basename(predicted_label))
        except FileNotFoundError:
            print("Error: Image file not found.")

"""
.......................................................................
                    Main Function 
........................................................................

"""
image_folders = [r"C:/Users/Dell/Music/CP_NP/urdu_alphabet_dataset/S",
                r"C:/Users/Dell/Music/CP_NP/urdu_alphabet_dataset/R",
                r"C:/Users/Dell/Music/CP_NP/urdu_alphabet_dataset/L",
                r"C:/Users/Dell/Music/CP_NP/urdu_alphabet_dataset/Z"]
classifier = ImageClassifier(image_folders)

classifier.load_images()

classifier.train_test_split()
classifier.train_model()
classifier.evaluate()

new_img_path = r"C:/Users/Dell/Music/CP_NP/test_image2.jpg"
classifier.predict(new_img_path)