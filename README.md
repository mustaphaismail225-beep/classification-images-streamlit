# Classification d'images avec Streamlit (CNN – CIFAR-10)

🔗 **Application en ligne :** https://classification-images-soufa.streamlit.app

## Capture d'écran

[Capture de l'application](capture.PNG)

## Explication

Nous avons déployé un réseau de neurones convolutif (CNN) entraîné sur le jeu de données CIFAR-10. Il reconnaît 10 classes d'images (avion, voiture, bateau, chat, etc.) avec une précision d'environ 69 % sur les données de test.
Dans l'application, l'utilisateur téléverse une image (JPG ou PNG), la visualise, puis obtient la classe prédite, le score de confiance et un graphique des probabilités par classe.
Le modèle a des limites : les images sont réduites à 32×32 pixels, donc il se trompe parfois sur des photos éloignées de CIFAR-10.
On pourrait ensuite améliorer la précision avec le transfer learning (MobileNetV2), accepter plusieurs images à la fois (saisie par lots) et permettre de télécharger les résultats.

## Fichiers

- `streamlit_app.py` : application Streamlit
- `modele_cifar10.h5` : modèle entraîné
- `train_model.py` : code d'entraînement du modèle
- `requirements.txt` : bibliothèques nécessaires

## Lancer en local

pip install -r requirements.txt
streamlit run streamlit_app.py
