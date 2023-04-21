import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model(s)
model_svm = joblib.load('svm_model.pkl')
model_dl = joblib.load('dl_model.pkl')

# Define the topic labels
labels = ['politics', 'technology', 'entertainment', 'health', 'business', 'sports']

# Define the Streamlit app
def main():
    st.title("News Article Topic Classification")
    st.markdown("This app classifies news articles into different topics.")

    # Define the input form
    article_text = st.text_area("Enter the text of your article:")
    model_choice = st.selectbox("Choose a model:", ["SVM", "Deep Learning"])

    # Make predictions using the chosen model
    if st.button("Classify"):
        if model_choice == "SVM":
            # Vectorize the input text
            vectorizer = joblib.load('svm_vectorizer.pkl')
            X = vectorizer.transform([article_text])

            # Make predictions using the SVM model
            y_pred = model_svm.predict(X)
            topic = labels[y_pred[0]]
        else:
            # Vectorize the input text
            tokenizer = joblib.load('dl_tokenizer.pkl')
            max_length = joblib.load('dl_max_length.pkl')
            X = tokenizer.texts_to_sequences([article_text])
            X = pad_sequences(X, maxlen=max_length, padding='post')

            # Make predictions using the deep learning model
            y_pred = model_dl.predict(X)
            topic = labels[np.argmax(y_pred)]

        # Show the predicted topic
        st.success("The predicted topic is: {}".format(topic))

if __name__ == '__main__':
    main()
