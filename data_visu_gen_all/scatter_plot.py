#!/usr/bin/env python3

import matplotlib
matplotlib.use("Agg")

import os
import re
import sys
import matplotlib.pyplot as plt
from parser import read_csv_file, get_numeric_features

OUTPUT_DIR = "files/scatter"
HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

# Transforme une chaîne de caractères en un nom de fichier sûr,
# en remplaçant les espaces et les caractères spéciaux par des underscores
# et en supprimant les caractères non alphanumériques
def sanitize_filename(name):
    name = name.strip().lower()
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")

# extrait les valeurs de deux caractéristiques numériques pour chaque maison,
# en vérifiant que les données sont valides
# et en les regroupant dans une structure adaptée pour la génération des scatter plots
def extract_two_features(rows, feature_x, feature_y):
    grouped = {
        "Gryffindor": {"x": [], "y": []},
        "Hufflepuff": {"x": [], "y": []},
        "Ravenclaw": {"x": [], "y": []},
        "Slytherin": {"x": [], "y": []}
    }

    for row in rows:
        x_value = row[feature_x].strip()
        y_value = row[feature_y].strip()
        house = row["Hogwarts House"].strip()

        if x_value == "" or y_value == "" or house == "":
            continue
        if house not in grouped:
            continue

        try:
            grouped[house]["x"].append(float(x_value))
            grouped[house]["y"].append(float(y_value))
        except ValueError:
            continue

    return grouped

# Vérifie que les données regroupées par maison contiennent au moins une paire de valeurs pour au moins une maison,
# afin de décider si le scatter plot peut être généré ou s'il doit être ignoré en raison de l'absence de données utilisables
def has_enough_data(grouped):
    for house in HOUSES:
        if grouped[house]["x"] and grouped[house]["y"]:
            return True
    return False

# Génère et sauvegarde un scatter plot pour deux caractéristiques données, en comparant les relations entre les maisons,
# et en utilisant une échelle cohérente basée sur les valeurs globales pour toutes les maisons, afin de faciliter la comparaison visuelle
def save_scatter_plot(feature_x, feature_y, grouped):
    plt.figure(figsize=(10, 6))

    for house in HOUSES:
        x_values = grouped[house]["x"]
        y_values = grouped[house]["y"]

        if x_values and y_values:
            plt.scatter(x_values, y_values, alpha=0.6, label=house)

    plt.title(f"{feature_x} vs {feature_y}")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()
    plt.tight_layout()

    safe_x = sanitize_filename(feature_x)
    safe_y = sanitize_filename(feature_y)
    output_path = os.path.join(OUTPUT_DIR, f"{safe_x}__vs__{safe_y}.png")

    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")
    return True


# Point d'entrée du programme : lit les données, extrait les valeurs de deux caractéristiques spécifiques
# et affiche un graphique de dispersion (scatter plot) pour visualiser la relation entre ces caractéristiques
# en fonction de la maison à laquelle appartiennent les étudiants, en sauvegardant les résultats dans des fichiers image
def main():
    if len(sys.argv) != 2:
        print("Usage: python3 scatter_plot.py dataset.csv")
        sys.exit(1)

    filename = sys.argv[1]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = read_csv_file(filename)
    features = sorted(get_numeric_features(rows))

    if len(features) < 2:
        print("Error: not enough numeric features found")
        sys.exit(1)

    generated_count = 0

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            feature_x = features[i]
            feature_y = features[j]

            grouped = extract_two_features(rows, feature_x, feature_y)

            if not has_enough_data(grouped):
                print(f"Skipped: {feature_x} vs {feature_y} (no usable data)")
                continue

            if save_scatter_plot(feature_x, feature_y, grouped):
                generated_count += 1

    print(f"\nGenerated {generated_count} scatter plot(s) in '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()