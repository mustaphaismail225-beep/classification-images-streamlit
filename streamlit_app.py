# Étape 2 : application Streamlit de classification d'images
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras

CLASSES = ["avion", "automobile", "oiseau", "chat", "cerf",
           "chien", "grenouille", "cheval", "bateau", "camion"]


# Charger le modèle une seule fois (gardé en mémoire)
@st.cache_resource
def charger_modele():
    return keras.models.load_model("modele_cifar10.h5")


# Prétraitement : même traitement que pendant l'entraînement
def pretraiter(image):
    image = image.convert("RGB")          # 3 canaux de couleur
    image = image.resize((32, 32))        # taille attendue par le modèle
    tableau = np.array(image) / 255.0     # normalisation 0-1
    return np.expand_dims(tableau, axis=0)  # forme (1, 32, 32, 3)


st.title("Classification d'images (CIFAR-10)")
st.write("Téléversez une image : avion, voiture, oiseau, chat, cerf, "
         "chien, grenouille, cheval, bateau ou camion.")

modele = charger_modele()

# 1. Téléversement de fichier
fichier = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if fichier is not None:
    # 2. Aperçu de l'image
    image = Image.open(fichier)
    st.image(image, caption="Image téléversée", width=300)

    # 3. Prétraitement + 4. Prédiction
    probas = modele.predict(pretraiter(image))[0]
    indice = int(np.argmax(probas))

    st.success(f"Classe prédite : {CLASSES[indice]}")

    # 5. Optionnel : score de confiance + graphique des probabilités
    st.write(f"Score de confiance : {probas[indice] * 100:.1f} %")
    df = pd.DataFrame({"Probabilité": probas}, index=CLASSES)
    st.bar_chart(df)
