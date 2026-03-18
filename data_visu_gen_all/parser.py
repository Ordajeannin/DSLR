import csv
import sys


# Lit un fichier CSV et organise les données en une liste de dictionnaires (une par ligne), en gérant les erreurs de lecture et les fichiers vides
def read_csv_file(filename):
    try:
        with open(filename, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
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

    return rows


# Vérifie si une colonne de données est numérique en essayant de convertir chaque valeur en float
def is_numeric_column(rows, column_name):
    found_numeric = False

    for row in rows:
        value = row[column_name].strip()
        if value == "":
            continue
        try:
            float(value)
            found_numeric = True
        except ValueError:
            return False
    return found_numeric


# Identifie les colonnes numériques et retourne une liste de leurs noms
def get_numeric_features(rows):
    headers = rows[0].keys()
    numeric_features = []

    for header in headers:
        if header in ["Index", "Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"]:
            continue
        if is_numeric_column(rows, header):
            numeric_features.append(header)

    return numeric_features


# Extrait les valeurs d'une caractéristique spécifique, en ignorant les valeurs vides et les lignes sans maison assignée
def get_feature_values(rows, feature_name):
    values = []
    houses = []

    for row in rows:
        value = row[feature_name].strip()
        house = row["Hogwarts House"].strip()

        if value == "" or house == "":
            continue

        try:
            values.append(float(value))
            houses.append(house)
        except ValueError:
            continue

    return values, houses


# Regroupe les valeurs d'une caractéristique par maison, en ignorant les valeurs vides
def get_values_by_house(rows, feature_name):
    grouped = {
        "Gryffindor": [],
        "Hufflepuff": [],
        "Ravenclaw": [],
        "Slytherin": []
    }

    for row in rows:
        value = row[feature_name].strip()
        house = row["Hogwarts House"].strip()

        if value == "" or house == "":
            continue

        try:
            grouped[house].append(float(value))
        except ValueError:
            continue

    return grouped