# Étape 1 : entraîner un CNN sur CIFAR-10 et le sauvegarder en .h5
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# A. Charger les données (incluses dans Keras)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# B. Normaliser les pixels (0-255 -> 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# C. Noms des classes
classes = ["avion", "automobile", "oiseau", "chat", "cerf",
           "chien", "grenouille", "cheval", "bateau", "camion"]

# D. Construire le CNN
model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

# E. Compiler
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# F. Entraîner
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# G. Sauvegarder
model.save("modele_cifar10.h5")
print("Modèle sauvegardé : modele_cifar10.h5")

# H. Tester le modèle rechargé avec predict()
modele_charge = keras.models.load_model("modele_cifar10.h5")
probas = modele_charge.predict(x_test[0:1])
print("Prédiction :", classes[np.argmax(probas)])
print("Vraie classe :", classes[y_test[0][0]])
