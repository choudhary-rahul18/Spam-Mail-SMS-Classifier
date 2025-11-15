import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import string

# --- Load Model and Vectorizer ---
try:
    tfidf = pickle.load(open('sms_classifier/Vectorizer.pkl', 'rb'))
    tfidf1 = pickle.load(open('sms_classifier/Vectorizer1.pkl', 'rb'))
    feature_extraction = pickle.load(open('mail_classifier/Vectorizer_spam_mail.pkl','rb'))
    mnb_model = pickle.load(open('sms_classifier/multi_naive_bayes_model.pkl', 'rb'))
    rf_model = pickle.load(open('sms_classifier/random_forest_model.pkl', 'rb'))
    lr_model = pickle.load(open('mail_classifier/logistic_reg_model.pkl', 'rb'))

except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    st.stop()


ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for s in text:
        if s.isalnum():
            y.append(s)

    text = y[:]
    y.clear()

    for s in text:
        if s not in stopwords.words('english') and s not in string.punctuation:
            y.append(s)

    text = y[:]
    y.clear()

    for s in text:
        y.append(ps.stem(s))

    return " ".join(y)





# Set page configuration for browser tab
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# Main title with an icon
st.title("📧 Email & SMS Spam Classifier")

model = st.selectbox(
    "Choose a classification model:",
    ("Naive Bayes", "Random Forest", "Logistic Regression")
)

# Use st.text_area for a larger input box suitable for messages
input_sms = st.text_area(
    "Enter a message below to check if it's spam or not",
    height=200,
    placeholder="Type or paste your message here..."
)

# Use st.button and make it "primary" to stand out
if st.button('Predict', type="primary"):

    # Add a check for empty input
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        selected_model = None
        if model == "Naive Bayes":
            selected_model = mnb_model
        elif model == "Random Forest":
            selected_model = rf_model
        elif model == "Logistic Regression":
            selected_model = lr_model

        if selected_model == mnb_model:
            # 1. Preprocess
            transform_sms = transform_text(input_sms)

            # 2. Vectorize
            # Note: tfidf.transform expects a list or iterable
            vector_input = tfidf1.transform([transform_sms])

            # 3. Predict
            # Note: model.predict expects the vectorized input
            result = selected_model.predict(vector_input)[0]  # Get the first prediction

        elif selected_model == rf_model:
            transform_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transform_sms])
            result = selected_model.predict(vector_input)[0]

        elif selected_model == lr_model:
            input_data_feature = feature_extraction.transform([input_sms])
            result = lr_model.predict(input_data_feature)

        # 4. Display
        if result == 0:
            st.success("✅ This message is **Not Spam**.")
        else:
            st.error("🚨 This message is **Spam**.")


st.sidebar.header("About")
st.sidebar.info(
    f"This app uses a Machine Learning model **({model})** "
    "and TF-IDF vectorization to classify messages as spam or not spam."
)
