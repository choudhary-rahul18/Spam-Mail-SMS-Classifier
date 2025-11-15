# 📧📱 Spam Mail & SMS Classifier

A machine learning project that builds and deploys two separate spam detection systems: one for SMS messages and one for emails.

Both classifiers are trained using **TF-IDF Vectorization** and evaluated with multiple machine learning models. The final models selected for deployment include:
* Logistic Regression
* Multinomial Naive Bayes
* Random Forest Classifier

A Streamlit application (`app.py`) loads the saved vectorizers and trained models to provide real-time spam predictions.

## 📂 Project Structure
```text
.
/mail_classifier
    ├── mail_data.csv
    ├── spam_mail_classifier.ipynb 
    ├── Vectorizer_spam_mail.pkl
    └── logistic_reg_model.pkl
    
/sms_classifier
    ├── spam.csv
    ├── sms_spam_classifier.ipynb
    ├── Vectorizer.pkl
    ├── Vectorizer1.pkl
    ├── random_forest_model.pkl
    └── multi_naive_bayes_model.pkl
    
/deployment
    └── app.py
    └── requirements.txt
```
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
## 🎯 Why Precision Was Our Priority

For spam detection tasks, predicting a normal message as spam (a **False Positive**) is more harmful than letting a real spam message through (a **False Negative**).

A false positive means:
* Important messages may get marked as spam.
* Emails or SMS that users are expecting might be blocked.
* The overall user experience gets worse.

Therefore, during model selection, **precision** was given higher importance than overall accuracy.

> **Precision:** Among all messages predicted as spam, how many were *actually* spam?

High precision ensures that the model marks a message as spam only when it is very confident, minimizing the number of false positives.

--
## 📊 Model Performance

| Model | Accuracy | Precision (Spam Class) |
| :--- | :--- | :--- |
| **Logistic Regression** | 0.9668 | **1.0000** |
| **Multinomial Naive Bayes** | 0.9729 | 0.9916 |
| **Random Forest Classifier**| 0.9700 | 0.9913 |

--

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
