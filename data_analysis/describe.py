#!/usr/bin/env python3

import sys
from parser import read_csv_file, get_numeric_columns
from stats import compute_describe
from display import print_describe

# Point d'entrée du programme : lit les données, calcule les statistiques descriptives et affiche les résultats
def main():
    if len(sys.argv) != 2:
        print("Usage: python3 describe.py dataset.csv")
        sys.exit(1)

    filename = sys.argv[1]

    headers, columns = read_csv_file(filename)
    numeric_data = get_numeric_columns(headers, columns)

    if not numeric_data:
        print("Error: no numeric columns found")
        sys.exit(1)

    column_names = list(numeric_data.keys())
    stats_order, result = compute_describe(numeric_data)
    print_describe(column_names, stats_order, result)


if __name__ == "__main__":
    main()