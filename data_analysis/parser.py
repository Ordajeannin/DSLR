import csv
import sys

# Vérifie si une colonne de données est numérique en essayant de convertir chaque valeur en float
def is_numeric_column(values):
    found_numeric = False

    for value in values:
        value = value.strip()
        if value == "":
            continue
        try:
            float(value)
            found_numeric = True
        except ValueError:
            return False
    return found_numeric


# Convertit une liste de chaînes de caractères en une liste de nombres à virgule flottante
# (en ignorant les valeurs vides)
def to_float_list(values):
    numbers = []

    for value in values:
        value = value.strip()
        if value == "":
            continue
        numbers.append(float(value))

    return numbers


# Lit un fichier CSV et organise les données en un dictionnaire de colonnes
def read_csv_file(filename):
    try:
        with open(filename, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: file '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not rows:
        print("Error: empty file")
        sys.exit(1)

    headers = rows[0]
    data_rows = rows[1:]

    columns = {}
    for i, header in enumerate(headers):
        columns[header] = []
        for row in data_rows:
            if i < len(row):
                columns[header].append(row[i])
            else:
                columns[header].append("")

    return headers, columns


# Identifie les colonnes numériques et convertit leurs valeurs en listes de floats
# (en ignorant les valeurs vides)
def get_numeric_columns(headers, columns):
    numeric_data = {}

    for header in headers:
        values = columns[header]
        if is_numeric_column(values):
            numeric_values = to_float_list(values)
            if len(numeric_values) > 0:
                numeric_data[header] = numeric_values

    return numeric_data