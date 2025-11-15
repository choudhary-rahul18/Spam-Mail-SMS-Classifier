# Spam Mail & SMS Classifier
A machine-learning project that builds and deploys two separate spam detection systems:
SMS Spam Classifier
Email Spam Classifier
Both classifiers are trained using TF-IDF Vectorization and multiple ML models. The final deployed models are:
* Logistic Regression
* Multinomial Naive Bayes
* Random Forest Classifier
  
A Streamlit app (app.py) loads the saved vectorizers + trained models to provide real-time predictions.

##📂 Project Structure
/mail_classifier
     ├── spam_mail_classifier.ipynb
     ├── mail_vectorizer.pkl
     ├── mail_model.pkl

/sms_classifier
     ├── sms_spam_classifier.ipynb
     ├── sms_vectorizer.pkl
     ├── sms_model.pkl

/deployment
     ├── app.py
