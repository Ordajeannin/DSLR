import math

#iter dans values et compte le nombre d'éléments
def count(values):
    c = 0
    for _ in values:
        c += 1
    return c


#iter dans values et additionne les éléments pour calculer la moyenne
def mean(values):
    total = 0.0
    n = 0

    for value in values:
        total += value
        n += 1

    if n == 0:
        return float("nan")
    return total / n


#iter dans values et calcule la somme des carrés des écarts à la moyenne pour calculer l'écart-type
def std(values, avg):
    n = count(values)
    if n == 0:
        return float("nan")

    variance_sum = 0.0
    for value in values:
        variance_sum += (value - avg) ** 2

    variance = variance_sum / n
    return math.sqrt(variance)


#iter dans values et trouve la valeur minimale
def minimum(values):
    if not values:
        return float("nan")

    current_min = values[0]
    for value in values[1:]:
        if value < current_min:
            current_min = value
    return current_min


#iter dans values et trouve la valeur maximale
def maximum(values):
    if not values:
        return float("nan")

    current_max = values[0]
    for value in values[1:]:
        if value > current_max:
            current_max = value
    return current_max


#trie les valeurs de la liste
def sort_values(values):
    return sorted(values)


#calcule le percentile p de la liste triée des valeurs
def percentile(sorted_values_list, p):
    n = count(sorted_values_list)

    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_values_list[0]

    pos = (n - 1) * p
    lower_index = int(math.floor(pos))
    upper_index = int(math.ceil(pos))

    if lower_index == upper_index:
        return sorted_values_list[lower_index]

    lower_value = sorted_values_list[lower_index]
    upper_value = sorted_values_list[upper_index]
    fraction = pos - lower_index

    return lower_value + (upper_value - lower_value) * fraction


#calcule les statistiques descriptives pour chaque colonne de données numériques 
#retourne un dictionnaire avec les résultats
def compute_describe(numeric_data):
    stats_order = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    result = {}

    for stat_name in stats_order:
        result[stat_name] = {}

    for column_name, values in numeric_data.items():
        sorted_vals = sort_values(values)
        avg = mean(values)

        result["Count"][column_name] = float(count(values))
        result["Mean"][column_name] = avg
        result["Std"][column_name] = std(values, avg)
        result["Min"][column_name] = minimum(values)
        result["25%"][column_name] = percentile(sorted_vals, 0.25)
        result["50%"][column_name] = percentile(sorted_vals, 0.50)
        result["75%"][column_name] = percentile(sorted_vals, 0.75)
        result["Max"][column_name] = maximum(values)

    return stats_order, result