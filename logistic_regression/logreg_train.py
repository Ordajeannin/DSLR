import csv
import json
import math
import sys
import matplotlib.pyplot as plt
import os

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

# Caractéristiques utilisées pour l'entraînement du modèle
FEATURES = [
    "Defense Against the Dark Arts",
    "Charms",
    "Ancient Runes",
    "Herbology",
    "Divination"]

# Hyperparamètres pour l'entraînement du modèle
LEARNING_RATE = 0.1
ITERATIONS = 5000
MODEL_PATH = "model.json"


# Remplace les valeurs manquantes (None) dans x_data
# par les valeurs de remplissage correspondantes dans fill_values
def fill_missing_values(x_data, fill_values):
    filled_data = []

    for row in x_data:
        new_row = []
        for j in range(len(row)):
            if row[j] is None:
                new_row.append(fill_values[j])
            else:
                new_row.append(row[j])
        filled_data.append(new_row)

    return filled_data


# Calcule la moyenne de chaque caractéristique en ignorant les valeurs manquantes (None)
def compute_feature_means(x_data):
    num_features = len(x_data[0])
    means = []

    for j in range(num_features):
        column = []
        for row in x_data:
            if row[j] is not None:
                column.append(row[j])

        mean = compute_mean(column)
        means.append(mean)

    return means


# Lit un fichier CSV et extrait les données pour les caractéristiques spécifiées
def load_dataset(path, features, training=True, fill_values=None):
    with open(path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    x_data = []
    y_data = []
    indices = []

    for row in rows:
        values = []

        for i, feature in enumerate(features):
            value = row[feature].strip()

            if value == "":
                if fill_values is not None:
                    values.append(fill_values[i])
                else:
                    values.append(None)
            else:
                values.append(float(value))

        x_data.append(values)
        indices.append(int(row["Index"]))

        if training:
            y_data.append(row["Hogwarts House"])

    if training:
        return indices, x_data, y_data
    return indices, x_data


# Calcule la moyenne d'une liste de valeurs
def compute_mean(values):
    return sum(values) / len(values)


# Calcule l'écart-type d'une liste de valeurs à partir de leur moyenne
def compute_std(values, mean):
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


# Calcule les paramètres de normalisation (moyennes et écarts-types)
# pour chaque caractéristique
def compute_normalization_params(x_data):
    num_features = len(x_data[0])
    means = []
    stds = []

    for j in range(num_features):
        column = [row[j] for row in x_data]
        mean = compute_mean(column)
        std = compute_std(column, mean)

        means.append(mean)
        stds.append(std if std != 0 else 1.0)

    return means, stds


# Normalise les données en soustrayant la moyenne
# et en divisant par l'écart-type pour chaque caractéristique
def normalize_dataset(x_data, means, stds):
    normalized = []

    for row in x_data:
        new_row = []
        for j in range(len(row)):
            new_row.append((row[j] - means[j]) / stds[j])
        normalized.append(new_row)

    return normalized


# Ajoute une colonne d'interception (valeur constante de 1)
# au début de chaque ligne de données pour permettre au modèle de calculer un biais
def add_intercept(x_data):
    result = []
    for row in x_data:
        result.append([1.0] + row)
    return result


# Calcule la fonction sigmoïde pour une valeur donnée, en gérant les cas de débordement numérique
def sigmoid(z):
    if z < -500:
        return 0.0
    if z > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


# Calcule le produit scalaire de deux vecteurs a et b
def dot(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


# Prédit la probabilité d'appartenance à la classe positive (target_house)
# pour une ligne de données x en utilisant les poids theta du modèle et la fonction sigmoïde
def predict_probability(x, theta):
    return sigmoid(dot(x, theta))


# Entraîne un modèle de régression logistique binaire
# pour différencier la target_house des autres classes en utilisant la descente de gradient
def train_one_vs_all(x_data, y_data, target_house, learning_rate, iterations):
    m = len(x_data)
    n = len(x_data[0])
    theta = [0.0] * n

    binary_y = []
    for house in y_data:
        binary_y.append(1.0 if house == target_house else 0.0)

    loss_history = []

    for iteration in range(iterations):
        gradients = [0.0] * n

        for i in range(m):
            prediction = predict_probability(x_data[i], theta)
            error = prediction - binary_y[i]

            for j in range(n):
                gradients[j] += error * x_data[i][j]

        for j in range(n):
            gradients[j] /= m
            theta[j] -= learning_rate * gradients[j]

        loss = compute_log_loss(x_data, binary_y, theta)
        loss_history.append(loss)

        if iteration % 500 == 0 or iteration == iterations - 1:
            print(f"[{target_house}] iteration {iteration:5d} | loss = {loss:.6f}")

    return theta, loss_history


# Sauvegarde les paramètres du modèle (caractéristiques, moyennes, écarts-types et poids)
# dans un fichier JSON pour une utilisation ultérieure lors de la prédiction
def save_model(path, features, means, stds, weights):
    model = {
        "features": features,
        "means": means,
        "stds": stds,
        "weights": weights
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=4)


# Calcule la log-loss (fonction de coût) pour évaluer les performances du modèle
# epsilon permet d eviter log(0)
def compute_log_loss(x_data, binary_y, theta):
    m = len(x_data)
    total = 0.0
    epsilon = 1e-15

    for i in range(m):
        prediction = predict_probability(x_data[i], theta)

        if prediction < epsilon:
            prediction = epsilon
        elif prediction > 1 - epsilon:
            prediction = 1 - epsilon

        total += binary_y[i] * math.log(prediction) + (1 - binary_y[i]) * math.log(1 - prediction)

    return -total / m


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


# Calcule l'exactitude des prédictions en comparant les maisons prédites avec les maisons réelles
def compute_accuracy(x_data, y_data, weights):
    correct = 0
    total = len(x_data)

    for i in range(total):
        predicted_house = predict_house(x_data[i], weights)
        if predicted_house == y_data[i]:
            correct += 1

    return correct / total



def plot_losses(all_losses, output_dir="files"):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for house, losses in all_losses.items():
        plt.plot(losses, label=house)

    plt.xlabel("Iterations")
    plt.ylabel("Log Loss")
    plt.title("Training Loss per House (One-vs-All)")
    plt.legend()

    output_path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Loss curves saved to {output_path}")



# Point d'entrée du programme : lit les données, normalise les caractéristiques,
# entraîne un modèle de régression logistique binaire pour chaque maison
# et sauvegarde les paramètres du modèle dans un fichier JSON
def main():
    if len(sys.argv) != 2:
        print("Usage: python3 logreg_train.py dataset_train.csv")
        sys.exit(1)

    dataset_path = sys.argv[1]

    indices, x_data, y_data = load_dataset(dataset_path, FEATURES, training=True)

    if not x_data:
        print("Error: no rows found")
        sys.exit(1)

    means = compute_feature_means(x_data)
    x_data = fill_missing_values(x_data, means)

    means, stds = compute_normalization_params(x_data)
    x_data = normalize_dataset(x_data, means, stds)
    x_data = add_intercept(x_data)

    weights = {}
    all_losses = {}

    for house in HOUSES:
        theta, loss_history = train_one_vs_all(
            x_data,
            y_data,
            house,
            LEARNING_RATE,
            ITERATIONS
        )
        weights[house] = theta
        all_losses[house] = loss_history

    train_accuracy = compute_accuracy(x_data, y_data, weights)
    print(f"\nTrain accuracy: {train_accuracy * 100:.2f}%")

    save_model(MODEL_PATH, FEATURES, means, stds, weights)
    print(f"Model saved to {MODEL_PATH}")
    plot_losses(all_losses)


if __name__ == "__main__":
    main()