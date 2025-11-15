# 📧📱 Spam Mail & SMS Classifier

A machine learning project that builds and deploys two separate spam detection systems: one for SMS messages and one for emails.

Both classifiers are trained using **TF-IDF Vectorization** and evaluated with multiple machine learning models. The final models selected for deployment include:
* Logistic Regression
* Multinomial Naive Bayes
* Random Forest Classifier

A Streamlit application (`app.py`) loads the saved vectorizers and trained models to provide real-time spam predictions.

## 📂 Project Structure
/mail_classifier
    ├── spam_mail_classifier.ipynb 
    ├── mail_vectorizer.pkl
    └── mail_model.pkl
    
/sms_classifier
    ├── sms_spam_classifier.ipynb
    ├── sms_vectorizer.pkl
    └── sms_model.pkl
    
/deployment
    └── app.py

---

## 🔍 File Descriptions

| File | Description |
| :--- | :--- |
| `sms_spam_classifier.ipynb` | Jupyter Notebook for training and testing SMS spam detection models. |
| `spam_mail_classifier.ipynb` | Notebook for training and testing Email spam classifiers. |
| `..._vectorizer.pkl` | Serialized TF-IDF vectorizer trained on the respective dataset. |
| `..._model.pkl` | Final trained model (e.g., Logistic Regression, Naive Bayes) chosen after evaluation. |
| `app.py` | The Streamlit application script used for deploying the classifier. |

---

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/spam-classifier.git](https://github.com/yourusername/spam-classifier.git)
    cd spam-classifier
    ```

2.  **Install Dependencies**
    (Ensure you have a `requirements.txt` file in your repository)
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit App**
    ```bash
    streamlit run deployment/app.py
    ```

---

## 🧠 Model Training Workflow

The training process, detailed in the `.ipynb` notebooks, follows these key steps:

1.  **Load Dataset**: Read the spam and ham (not spam) data.
2.  **Text Preprocessing**:
    * Convert text to lowercase.
    * Remove punctuation and special characters.
    * Remove stop words.
    * Apply stemming or lemmatization.
3.  **Vectorization**: Convert the cleaned text into numerical feature vectors using **TF-IDF**.
4.  **Train Models**: Train various classifiers, including:
    * Logistic Regression
    * Multinomial Naive Bayes
    * Random Forest
5.  **Evaluate**: Assess models using metrics like **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
6.  **Export**: Save the best-performing model and the TF-IDF vectorizer using `pickle` for later use in the app.

---

## 🚀 Deployment with Streamlit

The `app.py` script powers the web interface.

1.  It loads the pre-trained TF-IDF vectorizer and the saved model.
    ```python
    # Example from app.py
    tfidf_vectorizer = pickle.load(open('sms_vectorizer.pkl', 'rb'))
    model = pickle.load(open('sms_model.pkl', 'rb'))
    ```
2.  It takes text input from the user.
3.  The input text is preprocessed and transformed using the loaded vectorizer.
4.  The model predicts whether the input is **Spam** or **Not Spam** and displays the result.
