import sys
import os
import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
import joblib
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox, QHBoxLayout
from PyQt5.QtCore import Qt

# Constants for file paths
MODEL_PATH = "knn_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
DATA_PATH = r"Data\aidataset.csv"

# Preprocessing function
def preprocess_text(text):
    return " ".join(nltk.word_tokenize(text.lower()))

def is_valid_input(user_input):
    if not user_input.strip():
        return False
    if re.search(r'[\u0600-\u06FF]', user_input):  # Arabic characters
        return False
    if re.search(r'\d', user_input):  # Digits
        return False
    if len(user_input.split()) > 5:  # Limit to 5 words
        return False
    if any(len(word) >= 13 for word in user_input.split()):  # No word longer than 10 characters
        return False
    return True

# Training and saving model
def train_and_save_model():
    """
    Train and save a KNN model and vectorizer for future use.
    """
    # Load the data
    df = pd.read_csv(DATA_PATH)

    if 'Keyword' not in df.columns or 'Tip' not in df.columns:
        raise ValueError("The dataset must contain 'Keyword' and 'Tip' columns.")

    df['Keyword'] = df['Keyword'].str.lower()
    df['Processed_Keyword'] = df['Keyword'].apply(preprocess_text)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
    X = vectorizer.fit_transform(df['Processed_Keyword'])
    y = df['Keyword']

    # Train KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X, y)

    # Save the model and vectorizer
    joblib.dump(knn_classifier, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Model and vectorizer trained and saved successfully!")

# A* Search utility functions
def calculate_g_cost(word1, word2):

    return nltk.edit_distance(word1, word2)

def calculate_h_cost(user_vec, keyword_vec):
    similarity = cosine_similarity(user_vec, keyword_vec)[0][0]
    return 1 - similarity

def a_star_search(user_input, vectorizer, df):
    processed_input = preprocess_text(user_input)
    user_vec = vectorizer.transform([processed_input])

    open_list = []
    for index, keyword in enumerate(df['Keyword']):
        keyword_vec = vectorizer.transform([keyword])
        g_cost = calculate_g_cost(processed_input, keyword)
        h_cost = calculate_h_cost(user_vec, keyword_vec)
        f_cost = g_cost + h_cost
        open_list.append((f_cost, index, keyword))

    open_list.sort(key=lambda x: x[0])

    if open_list:
        best_match = open_list[0]
        predicted_keyword = df['Keyword'].iloc[best_match[1]]
        tip = df[df['Keyword'] == predicted_keyword]['Tip'].values[0]
        return predicted_keyword, tip
    else:
        raise ValueError("No suitable match found.")

def get_tip_a_star(user_input, vectorizer, df):
    if len(user_input) <= 1:
        raise ValueError("Input is too short. Please enter a more descriptive query.")

    try:
        keyword, tip = a_star_search(user_input, vectorizer, df)
        return keyword, tip
    except ValueError as e:
        raise ValueError(f"Error: {str(e)}")

class EcoFriendlyTipsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eco-Friendly Tips Generator")
        self.setGeometry(100, 100, 600, 400)

        self.initUI()

    def initUI(self):
        # إضافة ستايل للتطبيق
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial;
            }
            QLabel {
                color: #2c3e50;
                font-size: 14px;
                font-weight: bold;
            }
            QLineEdit, QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 5px;
                font-size: 18px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)

        self.layout = QVBoxLayout()

        # العنوان
        self.title_label = QLabel("\ud83c\udf0d Eco-Friendly Tips Generator \ud83c\udf31")
        self.title_label.setStyleSheet("font-size: 18px; color: #27ae60;")
        self.layout.addWidget(self.title_label, alignment=Qt.AlignCenter)

        # حقل الإدخال
        self.input_label = QLabel("Enter your eco-friendly query:")
        self.layout.addWidget(self.input_label)

        self.input_field = QLineEdit()
        self.layout.addWidget(self.input_field)

        # منطقة الإخراج
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setPlaceholderText("Tips will appear here...")
        self.layout.addWidget(self.output_area)

        # الأزرار
        button_layout = QHBoxLayout()
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.generate_tip)
        button_layout.addWidget(self.generate_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_fields)
        button_layout.addWidget(self.clear_button)

        self.layout.addLayout(button_layout)

        self.setLayout(self.layout)

    def generate_tip(self):
        user_input = self.input_field.text().strip()

        if not is_valid_input(user_input):
            QMessageBox.warning(self, "Invalid Input", "Please enter a meaningful English query.")
            return

        try:
            keyword, tip = get_tip_a_star(user_input, vectorizer, df)
            self.output_area.setText(f"Keyword: {keyword.capitalize()}\nEco-friendly Tip: {tip}")
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", "An unexpected error occurred. Please try again.")

    def clear_fields(self):
        self.input_field.clear()
        self.output_area.clear()

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Training model as no pre-trained model was found.")
        train_and_save_model()

    knn_classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    df = pd.read_csv(DATA_PATH)
    df['Keyword'] = df['Keyword'].str.lower()

    app = QApplication(sys.argv)
    window = EcoFriendlyTipsApp()
    window.show()
    sys.exit(app.exec_())
