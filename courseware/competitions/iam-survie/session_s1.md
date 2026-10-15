Modèle du jour : la **régression logistique** (`LogisticRegression` de
scikit-learn). Elle combine linéairement les variables : mettez-les à la même
échelle (`StandardScaler`) avant l'apprentissage, ajustez le scaler sur les seules
données d'entraînement, et gardez un jeu de validation pour estimer l'exactitude
avant de soumettre. Pour commencer simplement, utilisez les variables numériques
et remplacez les valeurs manquantes par la médiane.
