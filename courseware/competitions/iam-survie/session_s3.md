Modèle du jour : le **réseau de neurones** (`MLPClassifier`). Il exige des
variables imputées, encodées et mises à l'échelle. Choisissez le nombre et la
taille des couches cachées (`hidden_layer_sizes`) et utilisez l'**arrêt précoce**
(`early_stopping=True`) pour interrompre l'apprentissage quand le score de
validation cesse de progresser. Comparez enfin le réseau à la régression
logistique et à la forêt aléatoire des séances précédentes, sur le même jeu de
validation.
