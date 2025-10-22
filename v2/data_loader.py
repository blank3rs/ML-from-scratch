import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "muhammadtalharasool/simple-gender-classification",
    "gender.csv",
)

df.columns = df.columns.str.strip()

df = df[["Gender", "Age", "Height (cm)", "Income (USD)"]]

df["Gender"] = df["Gender"].str.strip().map({"male": 1, "female": 0})
df["Age"] /= df["Age"].max()
df["Height (cm)"] /= df["Height (cm)"].max()
df["Income (USD)"] /= df["Income (USD)"].max()

x = df[["Gender", "Age", "Height (cm)"]].values
y = df["Income (USD)"].values

print(df.head())
