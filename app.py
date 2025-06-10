import streamlit as st



import streamlit as st
from PIL import Image

st.title("Chargement et affichage d'image")

# 📤 Upload de fichier
uploaded_file = st.file_uploader("Choisissez une image...", type=["png"])

# 📸 Affichage de l'image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image chargée', use_column_width=True)
