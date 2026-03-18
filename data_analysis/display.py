import math

# Formate les valeurs flottantes pour l'affichage, en gérant les cas de NaN
def format_float(value):
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.2f}"

# Affiche les statistiques descriptives dans un format tabulaire
def print_describe(column_names, stats_order, result):
    # largeur de la première colonne (Count, Mean, etc.)
    first_col_width = max(len(stat) for stat in stats_order) + 2

    # calcul largeur dynamique pour chaque colonne
    col_widths = {}

    for col in column_names:
        max_width = len(col)

        for stat in stats_order:
            value = format_float(result[stat][col])
            if len(value) > max_width:
                max_width = len(value)

        col_widths[col] = max_width + 2  # padding

    # header
    header = " " * first_col_width
    for col in column_names:
        header += f"{col:>{col_widths[col]}}"
    print(header)

    # lignes
    for stat in stats_order:
        line = f"{stat:<{first_col_width}}"
        for col in column_names:
            value = format_float(result[stat][col])
            line += f"{value:>{col_widths[col]}}"
        print(line)