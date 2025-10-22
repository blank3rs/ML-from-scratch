import numpy as np
import os


def load_data() -> int:

    export_data = []
    data_path = os.path.join(os.path.dirname(__file__), "data.txt")
    with open(data_path, 'r') as file:
        data = file.read()
        lines = data.splitlines()

        for i in lines:
            line = i.split(":")
            export_data.append(
                [float(line[0])/100000, float(line[1])/100000])

    return np.array(export_data)


if __name__ == "__main__":
    print(load_data())
