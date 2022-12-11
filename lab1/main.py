import json

import capitalize as capitalize

file_data = "sat.json"
file_bace = "sat_stats.json"


def save_json(new_data: dict, file_name):
    """
    записываем пользователей в json
    """
    with open(file_name, 'w', encoding='utf-8') as file:
        data = json.dumps(new_data)
        file.write(data)


def read_json(file_name):
    """
    читам json файл с пользователями
    """
    with open(file_name, 'r', encoding='utf-8') as file:
        return json.load(file)


def main():
    database = read_json(file_bace)
    sat_model = []
    sat_generation = []
    ROS_sat = []
    ESA_sat = []
    NASA_sat = []
    JAXA_sat = []
    GLONASS = []
    YAMAL = []
    for i in database:
        if i["model"] == "Yamal":
            sat_model.append(i)
            sat_generation.append(i)
            YAMAL.append(i)
        if i["model"] == "Glonass":
            sat_model.append(i)
            sat_generation.append(i)
            GLONASS.append(i)
        if i["producer"] == "RosKosmos":
            sat_model.append(i)
            sat_generation.append(i)
            ROS_sat.append(i)
        if i["producer"] == "ESA":
            sat_model.append(i)
            sat_generation.append(i)
            ESA_sat.append(i)
        if i["producer"] == "NASA":
            sat_model.append(i)
            sat_generation.append(i)
            NASA_sat.append(i)
        if i["producer"] == "JAXA":
            sat_model.append(i)
            sat_generation.append(i)
            JAXA_sat.append(i)


def transformations(data):
    if data == 'ros_sat':
        data = ["sat", 'ROS']
    elif data == 'NASA_SAT':
        data = ["sat", 'US']
    elif data == 'ESA_SAT':
        data = ["sat", 'EU']
    elif data == 'JAXA_SAT':
        data = ["sat", 'JP']
    elif data == 'Yamal':
        data = ["sat", 'Yamal']
    elif data == 'Glonass':
        data = ["sat", 'Glonass']
    return data


# print(read_json())H
def out():
    print("| model | sat_generetion "
          "|\n| ROS_sat | NASA_SAT | ESA_SAT | JAXA_SAT |\n| Yamal | Glonass |")
    level = input("Выберете уровень: ")
    print("\nСлоты:")
    data = read_json(file_bace)[0][level]
    for i in data:
        print(i)
    slot = input("Выберете слот: ")
    print()
    if slot == "value":
        print(">= | = | <=")
        sign = input("Знак - ")
    else:
        sign = '='
    meaning = input("Значение - ")
    if level == "sat_generetion" or level == "model":
        search_in_main_class(level, slot, sign, meaning)
    else:
        search_in_inherited_class(level, slot, sign, meaning)


def search_in_main_class(level, slot, sign, meaning):
    database = read_json(file_bace)
    if sign == '=':
        for i in database:
            if slot == 'value':
                if i[slot] == int(meaning):
                    where(i, level, slot, meaning)
            elif i[slot] == meaning:
                where(i, level, slot, meaning)
    elif sign == '>=' and slot == 'value':
        for i in database:
            if i[slot] >= int(meaning):
                where(i, level, slot, meaning)
    elif sign == '<=' and slot == 'value':
        for i in database:
            if i[slot] <= int(meaning):
                where(i, level, slot, meaning)


def search_in_inherited_class(level, slot, sign, meaning):
    level = transformations(level)
    database = read_json(file_bace)
    if sign == '=':
        for i in database:
            if slot == 'value':
                if i[level[0]] == level[1] and i[slot] == int(meaning):
                    where(i, level, slot, meaning)
            elif i[level[0]] == level[1] and i[slot] == meaning:
                where(i, level, slot, meaning)
    elif sign == '>=' and slot == 'value':
        for i in database:
            if i[level[0]] == level[1] and i[slot] >= int(meaning):
                where(i, level, slot, meaning)
    elif sign == '<=' and slot == 'value':
        for i in database:
            if i[level[0]] == level[1] and i[slot] <= int(meaning):
                where(i, level, slot, meaning)


def where(data, level, slot, meaning):
    print()
    for i in data:
        print(i, ' - ', data[i], end=' | ')
    print()
    base = []
    if data["sat"] == 'YAMAL':
        base.append("SAT type -> YAMAL")
    if data["sat"] == 'GLONASS':
        base.append("SAT type -> GLONASS")
    if data["sat"] == 'ROS':
        base.append("COUNTRY OF ORIGIN -> RUSSIA")
    if data["sat"] == 'ESA':
        base.append("COUNTRY OF ORIGIN -> EUROPIAN UNION")
    if data["sat"] == 'NASA':
        base.append("COUNTRY OF ORIGIN -> USA")
    if data["sat"] == 'JAXA':
        base.append("COUNTRY OF ORIGIN -> JAPAN")
    for i in base:
        print(i, end=' | ')
    print()
    print("=" * 50)
    # for i in data_json:
    #     if data_json[i] != "False":
    #         print(data_json[i])


if __name__ == '__main__':
    out()