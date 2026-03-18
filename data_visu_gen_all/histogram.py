#!/usr/bin/env python3

import matplotlib
matplotlib.use("Agg")

import os
import re
import sys
import matplotlib.pyplot as plt
from parser import read_csv_file, get_numeric_features, get_values_by_house

OUTPUT_DIR = "files/histogram"
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

# Calcule le minimum et le maximum global parmi toutes les maisons pour une caractéristique donnée,
# afin d'assurer une échelle cohérente pour les histogrammes
def get_global_min_max(grouped_values):
    all_values = []

    for house in HOUSES:
        all_values.extend(grouped_values[house])

    if not all_values:
        return None, None

    min_value = all_values[0]
    max_value = all_values[0]

    for value in all_values[1:]:
        if value < min_value:
            min_value = value
        if value > max_value:
            max_value = value

    return min_value, max_value

# Vérifie que les données regroupées par maison contiennent au moins une valeur pour au moins une maison,
# afin de décider si l'histogramme peut être généré ou s'il doit être ignoré en raison de l'absence de données utilisables
def has_enough_data(grouped_values):
    for house in HOUSES:
        if grouped_values[house]:
            return True
    return False


# Génère et sauvegarde un histogramme pour une caractéristique donnée, en comparant les distributions entre les maisons,
# et en utilisant une échelle cohérente basée sur les valeurs globales pour toutes les maisons, afin de faciliter la comparaison visuelle
def save_histogram(feature, grouped_values):
    min_value, max_value = get_global_min_max(grouped_values)
    if min_value is None or max_value is None:
        return False

    plt.figure(figsize=(10, 6))

    for house in HOUSES:
        values = grouped_values[house]
        if values:
            plt.hist(values, bins=30, alpha=0.5, density=True, label=house)

    if min_value != max_value:
        plt.xlim(min_value, max_value)

    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    safe_name = sanitize_filename(feature)
    output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")

    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")
    return True


# Point d'entrée du programme : lit le dataset, identifie les caractéristiques numériques, prépare les données
# et génère un histogramme pour chaque caractéristique numérique, en comparant les distributions entre les maisons de Poudlard,
# et en sauvegardant les résultats dans des fichiers image
def main():
    if len(sys.argv) != 2:
        print("Usage: python3 histogram.py dataset.csv")
        sys.exit(1)

    filename = sys.argv[1]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = read_csv_file(filename)
    features = sorted(get_numeric_features(rows))

    if not features:
        print("Error: no numeric features found")
        sys.exit(1)

    generated_count = 0

    for feature in features:
        grouped_values = get_values_by_house(rows, feature)

        if not has_enough_data(grouped_values):
            print(f"Skipped: {feature} (no usable data)")
            continue

        if save_histogram(feature, grouped_values):
            generated_count += 1

    print(f"\nGenerated {generated_count} histogram(s) in '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()