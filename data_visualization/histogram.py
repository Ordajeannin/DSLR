#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
from parser import read_csv_file, get_numeric_features, get_values_by_house

# FEATURE_NAME = "Arithmancy"
FEATURE_NAME = "Astronomy"
# FEATURE_NAME = "Herbology"
# FEATURE_NAME = "Defense Against the Dark Arts"
# FEATURE_NAME = "Divination"
# FEATURE_NAME = "Muggle Studies"
# FEATURE_NAME = "Ancient Runes"
# FEATURE_NAME = "History of Magic"
# FEATURE_NAME = "Transfiguration"
# FEATURE_NAME = "Potions"
# FEATURE_NAME = "Care of Magical Creatures"
# FEATURE_NAME = "Charms"
# FEATURE_NAME = "Flying"



# Point d'entrée du programme :
# lit les données, identifie les caractéristiques numériques, 
# vérifie que la caractéristique choisie est présente, regroupe les valeurs de cette caractéristique par maison
# et affiche un histogramme pour comparer la distribution de cette caractéristique entre les différentes maisons
def main():
    if len(sys.argv) != 2:
        print("Usage: python3 histogram.py dataset_train.csv")
        sys.exit(1)

    filename = sys.argv[1]
    rows = read_csv_file(filename)
    features = get_numeric_features(rows)

    
    if FEATURE_NAME not in features:
        print(f"Error: feature '{FEATURE_NAME}' not found")
        sys.exit(1)

    grouped = get_values_by_house(rows, FEATURE_NAME)

    plt.hist(grouped["Gryffindor"], bins=30, alpha=0.5, label="Gryffindor")
    plt.hist(grouped["Hufflepuff"], bins=30, alpha=0.5, label="Hufflepuff")
    plt.hist(grouped["Ravenclaw"], bins=30, alpha=0.5, label="Ravenclaw")
    plt.hist(grouped["Slytherin"], bins=30, alpha=0.5, label="Slytherin")

    plt.title(f"Histogram of {FEATURE_NAME}")
    plt.xlabel(FEATURE_NAME)
    plt.ylabel("Number of students")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()