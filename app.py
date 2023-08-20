import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def predict_sentiment_and_recommendation(comment):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)
    
    comment_encoded = tokenizer.encode_plus(comment, truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        logits = model(**comment_encoded)[0]
    
    prediction = torch.argmax(logits, dim=1)[0].item() + 1

    if prediction == 1:
        star_rating = "1 estrella"
        recommendation = "Mejorar la calidad del servicio al cliente."
    elif prediction == 2:
        star_rating = "2 estrellas"
        recommendation = "Evalúa posibles mejoras en la experiencia del cliente."
    elif prediction == 3:
        star_rating = "3 estrellas"
        recommendation = "Considera realizar ajustes para mejorar la experiencia."
    elif prediction == 4:
        star_rating = "4 estrellas"
        recommendation = "Sigue proporcionando un buen servicio y calidad."
    else:
        star_rating = "5 estrellas"
        recommendation = "Sigue proporcionando un excelente servicio y calidad."

    return star_rating, recommendation

st.set_page_config(
    page_title="Data Hunters",
    page_icon="src\Huntersb.png",
)

st.title('Calificación de comentarios con Data Hunters')
st.write('Ingrese un comentario para obtener una predicción de sentimiento.')

comment = st.text_area("Ingresa tu comentario:")

if st.button("Predecir Sentimiento"):
    star_rating, recommendation = predict_sentiment_and_recommendation(comment)
    st.write(f'Predicción de Sentimiento: {star_rating}')
    st.write(f'Recomendación: {recommendation}')