#!/usr/bin/env python3

import matplotlib
matplotlib.use("Agg")

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

OUTPUT_DIR = "files/pair_plot"
OUTPUT_FILE = "pair_plot.png"

EXCLUDED_COLUMNS = [
    "Index",
    "Hogwarts House",
    "First Name",
    "Last Name",
    "Birthday",
    "Best Hand"
]

# Identifie les caractéristiques numériques dans le DataFrame, en excluant les colonnes spécifiées 
# et en vérifiant que les valeurs peuvent être converties en nombres
def get_numeric_features(df):
    numeric_features = []

    for column in df.columns:
        if column in EXCLUDED_COLUMNS:
            continue

        converted = pd.to_numeric(df[column], errors="coerce")

        if converted.notna().sum() > 0 and converted.notna().sum() == df[column].replace("", pd.NA).dropna().shape[0]:
            numeric_features.append(column)

    return numeric_features

# Point d'entrée du programme : lit le dataset, identifie les caractéristiques numériques, prépare les données
# et génère le pair plot pour comparer les distributions
# et les relations entre les caractéristiques numériques selon les maisons de Poudlard, en sauvegardant le résultat dans un fichier image
def main():
    if len(sys.argv) != 2:
        print("Usage: python3 pair_plot.py dataset.csv")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: file '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if df.empty:
        print("Error: empty dataset")
        sys.exit(1)

    if "Hogwarts House" not in df.columns:
        print("Error: 'Hogwarts House' column not found")
        sys.exit(1)

    numeric_features = sorted(get_numeric_features(df))

    if not numeric_features:
        print("Error: no numeric features found")
        sys.exit(1)

    selected_columns = ["Hogwarts House"] + numeric_features
    plot_df = df[selected_columns].copy()

    for feature in numeric_features:
        plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce")

    plot_df = plot_df.dropna()

    if plot_df.empty:
        print("Error: no complete rows available after dropping missing values")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    grid = sns.pairplot(
        plot_df,
        hue="Hogwarts House",
        diag_kind="hist",
        corner=False
    )

    grid.figure.savefig(os.path.join(OUTPUT_DIR, OUTPUT_FILE))
    plt.close("all")

    print(f"Saved: {os.path.join(OUTPUT_DIR, OUTPUT_FILE)}")


if __name__ == "__main__":
    main()