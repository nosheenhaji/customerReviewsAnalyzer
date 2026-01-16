import streamlit as st
import pickle
with open('logistic_model.pkl', 'rb') as file:
    data = pickle.load(file)

model = data['model']
vectorizer = data['vectorizer']
st.title("Customer Review Sentiment Prediction")

# User input
user_review = st.text_input("Enter your review here:")

# Button for prediction
if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a review!")
    else:
        # Vectorize the user review
        review_vector = vectorizer.transform([user_review])
        # Predict sentiment
        prediction = model.predict(review_vector)[0]
        
        # Agar aapke labels 0,1,2 hain
        label_map = {0:"negative", 1:"neutral", 2:"positive"}
        st.success(f"Predicted Sentiment: {label_map[prediction]}")
