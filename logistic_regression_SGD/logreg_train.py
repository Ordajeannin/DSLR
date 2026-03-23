import csv
import json
import math
import sys
import matplotlib.pyplot as plt
import os
import random

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

# Caractéristiques utilisées pour l'entraînement du modèle
FEATURES = [
    "Defense Against the Dark Arts",
    "Charms",
    "Ancient Runes",
    "Herbology",
    "Divination"]

# Hyperparamètres pour l'entraînement du modèle
LEARNING_RATE = 0.01
ITERATIONS = 1000
MODEL_PATH = "modelSDG.json"


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



# Crée le dossier de sortie s'il n'existe pas
def ensure_output_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Sauvegarde une image contenant toutes les courbes de loss
def plot_losses(all_losses, output_dir="files/losses"):
    ensure_output_dir(output_dir)

    plt.figure(figsize=(10, 6))

    for house, losses in all_losses.items():
        plt.plot(losses, label=house)

    plt.xlabel("Iterations")
    plt.ylabel("Log Loss")
    plt.title("Training Loss per House (One-vs-All)")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(output_dir, "loss_curves_all_houses.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")


# Sauvegarde une image de loss par maison
def plot_losses_separate(all_losses, output_dir="files/losses"):
    ensure_output_dir(output_dir)

    for house, losses in all_losses.items():
        plt.figure(figsize=(8, 5))
        plt.plot(losses)
        plt.xlabel("Iterations")
        plt.ylabel("Log Loss")
        plt.title(f"Training Loss - {house}")
        plt.grid(True)

        safe_house = house.lower().replace(" ", "_")
        output_path = os.path.join(output_dir, f"loss_{safe_house}.png")
        plt.savefig(output_path)
        plt.close()

        print(f"Saved: {output_path}")


# Sauvegarde un histogramme des probabilités prédites pour chaque classifieur
def plot_probability_distributions(x_data, weights, output_dir="files/proba"):
    ensure_output_dir(output_dir)

    for house, theta in weights.items():
        probabilities = []

        for x in x_data:
            prob = predict_probability(x, theta)
            probabilities.append(prob)

        plt.figure(figsize=(8, 5))
        plt.hist(probabilities, bins=30)
        plt.xlabel("Predicted probability")
        plt.ylabel("Number of students")
        plt.title(f"Probability Distribution - {house}")
        plt.grid(True)

        safe_house = house.lower().replace(" ", "_")
        output_path = os.path.join(output_dir, f"probabilities_{safe_house}.png")
        plt.savefig(output_path)
        plt.close()

        print(f"Saved: {output_path}")


# Construit une matrice de confusion sous forme de dictionnaire imbriqué
def compute_confusion_matrix(x_data, y_data, weights):
    matrix = {}

    for real_house in HOUSES:
        matrix[real_house] = {}
        for predicted_house in HOUSES:
            matrix[real_house][predicted_house] = 0

    for i in range(len(x_data)):
        real_house = y_data[i]
        predicted_house = predict_house(x_data[i], weights)
        matrix[real_house][predicted_house] += 1

    return matrix


# Affiche la matrice de confusion dans le terminal
def print_confusion_matrix(matrix):
    print("\nConfusion Matrix:")
    header = "Real \\ Pred".ljust(15)

    for house in HOUSES:
        header += house[:12].ljust(14)
    print(header)

    for real_house in HOUSES:
        row = real_house[:12].ljust(15)
        for predicted_house in HOUSES:
            row += str(matrix[real_house][predicted_house]).ljust(14)
        print(row)


# Sauvegarde une version visuelle simple de la matrice de confusion
def plot_confusion_matrix(matrix, output_dir="files"):
    ensure_output_dir(output_dir)

    data = []
    for real_house in HOUSES:
        row = []
        for predicted_house in HOUSES:
            row.append(matrix[real_house][predicted_house])
        data.append(row)

    plt.figure(figsize=(8, 6))
    plt.imshow(data, interpolation="nearest")
    plt.colorbar()

    plt.xticks(range(len(HOUSES)), HOUSES, rotation=45)
    plt.yticks(range(len(HOUSES)), HOUSES)
    plt.xlabel("Predicted House")
    plt.ylabel("Real House")
    plt.title("Confusion Matrix")

    for i in range(len(HOUSES)):
        for j in range(len(HOUSES)):
            plt.text(j, i, str(data[i][j]), ha="center", va="center")

    plt.tight_layout()

    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")



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


# Sauvegarde les courbes de loss dans des fichiers séparés pour chaque maison
def save_losses(path, all_losses):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_losses, f, indent=4)

    print(f"Saved: {path}")

def train_sgd(x_data, y_data, house_name):
    n_features = len(x_data[0])
    weights = [0.0] * n_features

    for iteration in range(ITERATIONS):
        combined = list(zip(x_data, y_data))
        random.shuffle(combined)

        for x, y in combined:
            z = sum(w * xi for w, xi in zip(weights, x))
            y_hat = sigmoid(z)

            error = y_hat - y

            for i in range(n_features):
                weights[i] -= LEARNING_RATE * error * x[i]
        
        if iteration % 500 == 0 or iteration == ITERATIONS -1:
            loss = compute_log_loss(x_data, y_data, weights)
            print(f"[{house_name}] iter {iteration:5d} | loss = {loss:.6f}")

    return weights

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

    models = {}

    for house in HOUSES:
        y_binary = [1 if y == house else 0 for y in y_data]
        weights = train_sgd(x_data, y_binary, house)
        models[house] = weights
        
    acc = compute_accuracy(x_data, y_data, models)
    print(f"training accuracy: {acc:.4f}")
    
    model = {
        "features": FEATURES,   # liste des caractéristiques
        "means": means,         # moyennes pour normalisation
        "stds": stds,           # écarts-types pour normalisation
        "weights": models       # dictionnaire des poids par maison
    }

    with open(MODEL_PATH, "w") as f:
        json.dump(model, f)


if __name__ == "__main__":
    main()