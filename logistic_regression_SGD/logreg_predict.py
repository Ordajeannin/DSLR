import csv
import json
import math
import sys

# Lit un fichier CSV et extrait les données pour les caractéristiques spécifiées
def load_dataset(path, features, means=None):
    with open(path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    indices = []
    x_data = []

    for row in rows:
        values = []

        for i, feature in enumerate(features):
            value = row[feature].strip()

            if value == "":
                if means is not None:
                    values.append(means[i])
                else:
                    values.append(0.0)
            else:
                values.append(float(value))

        indices.append(int(row["Index"]))
        x_data.append(values)

    return indices, x_data


# Normalise les données en soustrayant la moyenne et en divisant par l'écart-type pour chaque caractéristique
def normalize_dataset(x_data, means, stds):
    normalized = []

    for row in x_data:
        new_row = []
        for j in range(len(row)):
            std = stds[j] if stds[j] != 0 else 1.0
            new_row.append((row[j] - means[j]) / std)
        normalized.append(new_row)

    return normalized

# Ajoute une colonne d'interception (valeur constante de 1)
# au début de chaque ligne de données pour permettre au modèle de calculer un biais
def add_intercept(x_data):
    result = []
    for row in x_data:
        result.append([1.0] + row)
    return result


# Fonction sigmoïde pour convertir une valeur z en une probabilité entre 0 et 1
def sigmoid(z):
    if z < -500:
        return 0.0
    if z > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


# Calcule le produit scalaire entre deux vecteurs a et b
def dot(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

# Prédit la maison d'un étudiant en calculant la probabilité d'appartenance à chaque maison
# et en choisissant celle avec la probabilité la plus élevée
def predict_house(x, weights):
    best_house = None
    best_score = -1.0

    for house, theta in weights.items():
        score = sigmoid(dot(x, theta))
        if score > best_score:
            best_score = score
            best_house = house

    return best_house


# Sauvegarde les prédictions dans un fichier CSV avec les indices et les maisons prédites
def save_predictions(path, indices, predictions):
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Index", "Hogwarts House"])

        for i in range(len(indices)):
            writer.writerow([indices[i], predictions[i]])


# Calcule l'exactitude des prédictions en comparant les maisons prédites avec les maisons réelles
def compute_accuracy(x_data, y_data, weights):
    correct = 0
    total = len(x_data)

    for i in range(total):
        predicted_house = predict_house(x_data[i], weights)
        if predicted_house == y_data[i]:
            correct += 1

    return correct / total


# Point d'entrée du programme : lit les données, normalise les caractéristiques,
# prédit la maison pour chaque étudiant en utilisant les poids du modèle
# et sauvegarde les prédictions dans un fichier CSV
def main():
    if len(sys.argv) != 3:
        print("Usage: python3 logreg_predict.py dataset_test.csv model.json")
        sys.exit(1)

    dataset_path = sys.argv[1]
    model_path = sys.argv[2]

    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    features = model["features"]
    means = model["means"]
    stds = model["stds"]
    weights = model["weights"]

    indices, x_data = load_dataset(dataset_path, features, means)

    if not x_data:
        print("Error: no valid rows found")
        sys.exit(1)

    x_data = normalize_dataset(x_data, means, stds)
    x_data = add_intercept(x_data)

    predictions = []
    for row in x_data:
        predictions.append(predict_house(row, weights))

    save_predictions("housesSGD.csv", indices, predictions)
    print("Predictions saved to housesSGD.csv")


if __name__ == "__main__":
    main()