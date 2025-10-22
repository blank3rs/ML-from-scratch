import os

data_path = os.path.join(os.path.dirname(__file__), "data.txt")
with open(data_path, 'w') as file:
    for i in range(100000):
        file.writelines(f"{str(i)}:{str(i-1)}\n")
