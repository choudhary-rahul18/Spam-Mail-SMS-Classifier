# 📧📱 Spam Mail & SMS Classifier

A machine learning project that builds and deploys two separate spam detection systems: one for SMS messages and one for emails.

Both classifiers are trained using **TF-IDF Vectorization** and evaluated with multiple machine learning models. The final models selected for deployment include:
* Logistic Regression
* Multinomial Naive Bayes
* Random Forest Classifier

A Streamlit application (`app.py`) loads the saved vectorizers and trained models to provide real-time spam predictions.

## 📂 Project Structure
. ├── /mail_classifier │ ├── spam_mail_classifier.ipynb │ ├── mail_vectorizer.pkl │ └── mail_model.pkl │ ├── /sms_classifier │ ├── sms_spam_classifier.ipynb │ ├── sms_vectorizer.pkl │ └── sms_model.pkl │ ├── /deployment │ └── app.py │ └── /datasets ├── sms_dataset.csv └── mail_dataset.csv
