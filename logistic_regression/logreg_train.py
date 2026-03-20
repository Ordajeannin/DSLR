import csv
import json
import math
import sys
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio

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
    theta_history = []

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
        
        theta_history.append(theta[:])

        loss = compute_log_loss(x_data, binary_y, theta)
        loss_history.append(loss)

        if iteration % 500 == 0 or iteration == iterations - 1:
            print(f"[{target_house}] iteration {iteration:5d} | loss = {loss:.6f}")

    return theta, loss_history, theta_history


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



# Sauvegarde les courbes de loss dans des fichiers séparés pour chaque maison
def save_losses(path, all_losses):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_losses, f, indent=4)

    print(f"Saved: {path}")



def animate_house_learning(x_data, y_data, theta_history, target_house, output_dir="files"):
    ensure_output_dir(output_dir)

    frames_dir = os.path.join(output_dir, "frames")
    snapshots_dir = os.path.join(output_dir, "snapshots")

    ensure_output_dir(frames_dir)
    ensure_output_dir(snapshots_dir)

    frame_paths = []

    step = max(1, len(theta_history) // 80)

    for frame_idx, theta in enumerate(theta_history[::step]):
        positive_scores = []
        negative_scores = []

        for i in range(len(x_data)):
            z = dot(x_data[i], theta)
            if y_data[i] == target_house:
                positive_scores.append(z)
            else:
                negative_scores.append(z)

        plt.figure(figsize=(10, 5))
        plt.hist(negative_scores, bins=30, alpha=0.7, label=f"Not {target_house}")
        plt.hist(positive_scores, bins=30, alpha=0.7, label=target_house)
        plt.xlabel("Raw score z = θ·x")
        plt.ylabel("Number of students")
        plt.title(f"Learning evolution - {target_house} (frame {frame_idx})")
        plt.legend()
        plt.grid(True)

        frame_path = os.path.join(frames_dir, f"{target_house.lower()}_{frame_idx:03d}.png")
        plt.savefig(frame_path)
        plt.close()

        frame_paths.append(frame_path)

        # 🔥 Sauvegarde frame 0
        if frame_idx == 0:
            first_path = os.path.join(snapshots_dir, f"{target_house.lower()}_start.png")
            plt.figure(figsize=(10, 5))
            plt.hist(negative_scores, bins=30, alpha=0.7, label=f"Not {target_house}")
            plt.hist(positive_scores, bins=30, alpha=0.7, label=target_house)
            plt.title(f"{target_house} - START")
            plt.legend()
            plt.grid(True)
            plt.savefig(first_path)
            plt.close()

    # 🔥 Sauvegarde dernière frame
    last_theta = theta_history[-1]

    positive_scores = []
    negative_scores = []

    for i in range(len(x_data)):
        z = dot(x_data[i], last_theta)
        if y_data[i] == target_house:
            positive_scores.append(z)
        else:
            negative_scores.append(z)

    last_path = os.path.join(snapshots_dir, f"{target_house.lower()}_end.png")

    plt.figure(figsize=(10, 5))
    plt.hist(negative_scores, bins=30, alpha=0.7, label=f"Not {target_house}")
    plt.hist(positive_scores, bins=30, alpha=0.7, label=target_house)
    plt.title(f"{target_house} - END")
    plt.legend()
    plt.grid(True)
    plt.savefig(last_path)
    plt.close()

    # 🎬 GIF
    gif_path = os.path.join(output_dir, f"learning_{target_house.lower()}.gif")

    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, images, duration=0.15)

    print(f"Saved GIF: {gif_path}")
    print(f"Saved snapshots: {snapshots_dir}")
    for path in frame_paths:
        if os.path.exists(path):
            os.remove(path)



def plot_scores_and_probabilities(x_data, y_data, weights, output_dir="files/proba_by_class"):
    ensure_output_dir(output_dir)

    for house, theta in weights.items():
        positive_scores = []
        negative_scores = []
        positive_probs = []
        negative_probs = []

        for i in range(len(x_data)):
            z = dot(x_data[i], theta)
            p = sigmoid(z)

            if y_data[i] == house:
                positive_scores.append(z)
                positive_probs.append(p)
            else:
                negative_scores.append(z)
                negative_probs.append(p)

        plt.figure(figsize=(10, 5))
        plt.hist(negative_scores, bins=30, alpha=0.7, label=f"Not {house}")
        plt.hist(positive_scores, bins=30, alpha=0.7, label=house)
        plt.xlabel("Raw score z = θ·x")
        plt.ylabel("Number of students")
        plt.title(f"Score distribution - {house}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"scores_{house.lower()}.png"))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.hist(negative_probs, bins=30, alpha=0.7, label=f"Not {house}")
        plt.hist(positive_probs, bins=30, alpha=0.7, label=house)
        plt.xlabel("Predicted probability")
        plt.ylabel("Number of students")
        plt.title(f"Probability distribution by true class - {house}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"prob_by_class_{house.lower()}.png"))
        plt.close()




def plot_sigmoid_with_students(x_data, y_data, weights, target_house, output_dir="files/sigmoids"):
    ensure_output_dir(output_dir)

    theta = weights[target_house]

    z_values = []
    p_values = []
    colors = []

    for i in range(len(x_data)):
        z = dot(x_data[i], theta)
        p = sigmoid(z)
        z_values.append(z)
        p_values.append(p)

        if y_data[i] == target_house:
            colors.append("positive")
        else:
            colors.append("negative")

    z_curve = [x / 10.0 for x in range(-100, 101)]
    p_curve = [sigmoid(z) for z in z_curve]

    plt.figure(figsize=(10, 6))
    plt.plot(z_curve, p_curve, label="Sigmoid")

    pos_x = [z_values[i] for i in range(len(z_values)) if colors[i] == "positive"]
    pos_y = [p_values[i] for i in range(len(p_values)) if colors[i] == "positive"]

    neg_x = [z_values[i] for i in range(len(z_values)) if colors[i] == "negative"]
    neg_y = [p_values[i] for i in range(len(p_values)) if colors[i] == "negative"]

    plt.scatter(neg_x, neg_y, alpha=0.5, label=f"Not {target_house}")
    plt.scatter(pos_x, pos_y, alpha=0.5, label=target_house)

    plt.xlabel("Raw score z = θ·x")
    plt.ylabel("Probability")
    plt.title(f"How probabilities are computed - {target_house}")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(output_dir, f"sigmoid_{target_house.lower()}.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")


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
    all_thetas = {}

    for house in HOUSES:
        theta, loss_history, theta_history = train_one_vs_all(
            x_data,
            y_data,
            house,
            LEARNING_RATE,
            ITERATIONS
        )
        weights[house] = theta
        all_losses[house] = loss_history
        all_thetas[house] = theta_history

    for house in HOUSES:
        animate_house_learning(x_data, y_data, all_thetas[house], house)

    train_accuracy = compute_accuracy(x_data, y_data, weights)
    print(f"\nTrain accuracy: {train_accuracy * 100:.2f}%")

    plot_losses(all_losses)
    plot_losses_separate(all_losses)
    plot_probability_distributions(x_data, weights)

    confusion = compute_confusion_matrix(x_data, y_data, weights)
    print_confusion_matrix(confusion)
    plot_confusion_matrix(confusion)

    save_model(MODEL_PATH, FEATURES, means, stds, weights)
    print(f"Model saved to {MODEL_PATH}")
    save_losses("files/losses/loss_history.json", all_losses)
    plot_scores_and_probabilities(x_data, y_data, weights)


if __name__ == "__main__":
    main()