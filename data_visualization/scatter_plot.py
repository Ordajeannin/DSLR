#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
from parser import read_csv_file


FEATURE_X = "Arithmancy"
# FEATURE_X = "Astronomy"
# FEATURE_X = "Herbology"
# FEATURE_X = "Defense Against the Dark Arts"
# FEATURE_X = "Divination"
# FEATURE_X = "Muggle Studies"
# FEATURE_X = "Ancient Runes"
# FEATURE_X = "History of Magic"
# FEATURE_X = "Transfiguration"
# FEATURE_X = "Potions"
# FEATURE_X = "Care of Magical Creatures"
# FEATURE_X = "Charms"
# FEATURE_X = "Flying"


# FEATURE_Y = "Arithmancy"
FEATURE_Y = "Astronomy"
# FEATURE_Y = "Herbology"
# FEATURE_Y = "Defense Against the Dark Arts"
# FEATURE_Y = "Divination"
# FEATURE_Y = "Muggle Studies"
# FEATURE_Y = "Ancient Runes"
# FEATURE_Y = "History of Magic"
# FEATURE_Y = "Transfiguration"
# FEATURE_Y = "Potions"
# FEATURE_Y = "Care of Magical Creatures"
# FEATURE_Y = "Charms"
# FEATURE_Y = "Flying"



# Extrait les valeurs de deux caractéristiques spécifiques, en ignorant les valeurs vides et les lignes sans maison assignée
def extract_two_features(rows, feature_x, feature_y):
    x_values = []
    y_values = []
    houses = []

    for row in rows:
        x = row[feature_x].strip()
        y = row[feature_y].strip()
        house = row["Hogwarts House"].strip()

        if x == "" or y == "" or house == "":
            continue

        try:
            x_values.append(float(x))
            y_values.append(float(y))
            houses.append(house)
        except ValueError:
            continue

    return x_values, y_values, houses


# Point d'entrée du programme : 
# lit les données, extrait les valeurs de deux caractéristiques spécifiques 
# et affiche un graphique de dispersion (scatter plot) pour visualiser la relation entre ces caractéristiques 
# en fonction de la maison à laquelle appartiennent les étudiants
def main():
    if len(sys.argv) != 2:
        print("Usage: python3 scatter_plot.py dataset_train.csv")
        sys.exit(1)

    fileX = sys.argv[1]
    rows = read_csv_file(fileX)

    x_values, y_values, houses = extract_two_features(rows, FEATURE_X, FEATURE_Y)

    colors = {
        "Gryffindor": "red",
        "Hufflepuff": "yellow",
        "Ravenclaw": "blue",
        "Slytherin": "green"
    }

    for house in colors:
        x_house = []
        y_house = []

        for i in range(len(houses)):
            if houses[i] == house:
                x_house.append(x_values[i])
                y_house.append(y_values[i])

        plt.scatter(x_house, y_house, label=house, alpha=0.6)

    plt.title(f"{FEATURE_X} vs {FEATURE_Y}")
    plt.xlabel(FEATURE_X)
    plt.ylabel(FEATURE_Y)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()