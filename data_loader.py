import numpy as np


def load_data() -> int:

    export_data = []
    with open("data.txt", 'r') as file:
        data = file.read()
        lines = data.splitlines()

        for i in lines:
            line = i.split(":")
            export_data.append([float(line[0]), float(line[1])])

    return np.array(export_data)


if __name__ == "__main__":
    print(load_data())
