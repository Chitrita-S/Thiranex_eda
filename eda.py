import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS PROJECT")
print("=" * 70)

print("\nLoading Titanic dataset...")

df = sns.load_dataset("titanic")

print("\nDataset loaded successfully.")

print("\n" + "=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

print("\nNumerical Summary:")
print(df.describe())

print("\nCategorical Summary:")
print(df.describe(include="object"))

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing_values = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
).round(2)

missing_report = pd.DataFrame({
    "Missing Values": missing_values,
    "Missing Percentage": missing_percentage
})

missing_report = missing_report.sort_values(
    by="Missing Values",
    ascending=False
)

print(missing_report)

missing_report.to_csv(
    "missing_value_report.csv"
)

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

print("\n" + "=" * 70)
print("SURVIVAL ANALYSIS")
print("=" * 70)

survival_counts = df["survived"].value_counts()

print("\nSurvival Counts:")
print(survival_counts)

survival_percentage = (
    df["survived"]
    .value_counts(normalize=True)
    * 100
).round(2)

print("\nSurvival Percentage:")
print(survival_percentage)

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="survived"
)

plt.title("Survival Distribution")

plt.xlabel("Survival Status")

plt.ylabel("Number of Passengers")

plt.xticks(
    [0, 1],
    ["Did Not Survive", "Survived"]
)

plt.tight_layout()

plt.savefig(
    "01_survival_distribution.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="sex"
)

plt.title("Passenger Distribution by Gender")

plt.xlabel("Gender")

plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig(
    "02_gender_distribution.png",
    dpi=300
)

plt.show()

gender_survival = pd.crosstab(
    df["sex"],
    df["survived"],
    normalize="index"
) * 100

print("\n" + "=" * 70)
print("SURVIVAL RATE BY GENDER")
print("=" * 70)

print(
    gender_survival.round(2)
)


plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="sex",
    y="survived"
)

plt.title("Survival Rate by Gender")

plt.xlabel("Gender")

plt.ylabel("Survival Rate")

plt.tight_layout()

plt.savefig(
    "03_survival_by_gender.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="class"
)

plt.title("Passenger Distribution by Class")

plt.xlabel("Passenger Class")

plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig(
    "04_passenger_class_distribution.png",
    dpi=300
)

plt.show()

class_survival = pd.crosstab(
    df["class"],
    df["survived"],
    normalize="index"
) * 100

print("\n" + "=" * 70)
print("SURVIVAL RATE BY PASSENGER CLASS")
print("=" * 70)

print(
    class_survival.round(2)
)


plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="class",
    y="survived"
)

plt.title("Survival Rate by Passenger Class")

plt.xlabel("Passenger Class")

plt.ylabel("Survival Rate")

plt.tight_layout()

plt.savefig(
    "05_survival_by_class.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="age",
    bins=30,
    kde=True
)

plt.title("Age Distribution of Passengers")

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "06_age_distribution.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="survived",
    y="age"
)

plt.title("Age Distribution by Survival Status")

plt.xlabel("Survival Status")

plt.ylabel("Age")

plt.xticks(
    [0, 1],
    ["Did Not Survive", "Survived"]
)

plt.tight_layout()

plt.savefig(
    "07_age_vs_survival.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="fare",
    bins=30,
    kde=True
)

plt.title("Fare Distribution")

plt.xlabel("Fare")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "08_fare_distribution.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="survived",
    y="fare"
)

plt.title("Fare Distribution by Survival Status")

plt.xlabel("Survival Status")

plt.ylabel("Fare")

plt.xticks(
    [0, 1],
    ["Did Not Survive", "Survived"]
)

plt.tight_layout()

plt.savefig(
    "09_fare_vs_survival.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    alpha=0.7
)

plt.title("Age vs Fare")

plt.xlabel("Age")

plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    "10_age_vs_fare.png",
    dpi=300
)

plt.show()

df["family_size"] = (
    df["sibsp"] +
    df["parch"] +
    1
)

print("\n" + "=" * 70)
print("FAMILY SIZE ANALYSIS")
print("=" * 70)

family_survival = (
    df.groupby("family_size")["survived"]
    .mean()
    .sort_index()
)

print(
    family_survival.round(3)
)


plt.figure(figsize=(9, 5))

sns.lineplot(
    x=family_survival.index,
    y=family_survival.values,
    marker="o"
)

plt.title("Survival Rate by Family Size")

plt.xlabel("Family Size")

plt.ylabel("Survival Rate")

plt.tight_layout()

plt.savefig(
    "11_survival_by_family_size.png",
    dpi=300
)

plt.show()

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

numeric_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "adult_male",
    "alone"
]

numeric_df = df[numeric_columns].copy()

correlation_matrix = numeric_df.corr()

print(
    correlation_matrix.round(2)
)

correlation_matrix.to_csv(
    "correlation_matrix.csv"
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "12_correlation_heatmap.png",
    dpi=300
)

plt.show()

print("\n" + "=" * 70)
print("GROUPED ANALYSIS")
print("=" * 70)

grouped_analysis = (
    df.groupby(
        ["class", "sex"]
    )
    .agg(
        Passengers=("survived", "count"),
        Survival_Rate=("survived", "mean"),
        Average_Age=("age", "mean"),
        Average_Fare=("fare", "mean")
    )
)

grouped_analysis["Survival_Rate"] = (
    grouped_analysis["Survival_Rate"] * 100
).round(2)

grouped_analysis["Average_Age"] = (
    grouped_analysis["Average_Age"].round(2)
)

grouped_analysis["Average_Fare"] = (
    grouped_analysis["Average_Fare"].round(2)
)

print(grouped_analysis)


grouped_analysis.to_csv(
    "grouped_analysis_report.csv"
)

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

overall_survival = (
    df["survived"].mean() * 100
)

print(
    f"\n1. Overall survival rate: "
    f"{overall_survival:.2f}%"
)

gender_rates = (
    df.groupby("sex")["survived"]
    .mean() * 100
)

highest_gender = gender_rates.idxmax()

print(
    f"2. Higher observed survival rate by gender: "
    f"{highest_gender} "
    f"({gender_rates.max():.2f}%)"
)

class_rates = (
    df.groupby("class")["survived"]
    .mean() * 100
)

highest_class = class_rates.idxmax()

print(
    f"3. Higher observed survival rate by class: "
    f"{highest_class} "
    f"({class_rates.max():.2f}%)"
)

survived_age = (
    df[df["survived"] == 1]["age"]
    .mean()
)

not_survived_age = (
    df[df["survived"] == 0]["age"]
    .mean()
)

print(
    f"4. Average age of survivors: "
    f"{survived_age:.2f}"
)

print(
    f"5. Average age of non-survivors: "
    f"{not_survived_age:.2f}"
)

survived_fare = (
    df[df["survived"] == 1]["fare"]
    .mean()
)

not_survived_fare = (
    df[df["survived"] == 0]["fare"]
    .mean()
)

print(
    f"6. Average fare of survivors: "
    f"{survived_fare:.2f}"
)

print(
    f"7. Average fare of non-survivors: "
    f"{not_survived_fare:.2f}"
)

survival_correlation = (
    correlation_matrix["survived"]
    .drop("survived")
    .abs()
    .sort_values(
        ascending=False
    )
)

print(
    "\n8. Variables with strongest absolute "
    "correlation with survival:"
)

print(
    survival_correlation.head(5)
)

insights = {
    "Overall Survival Rate (%)":
        round(overall_survival, 2),

    "Highest Survival Gender":
        highest_gender,

    "Highest Gender Survival Rate (%)":
        round(gender_rates.max(), 2),

    "Highest Survival Class":
        str(highest_class),

    "Highest Class Survival Rate (%)":
        round(class_rates.max(), 2),

    "Average Age - Survivors":
        round(survived_age, 2),

    "Average Age - Non-Survivors":
        round(not_survived_age, 2),

    "Average Fare - Survivors":
        round(survived_fare, 2),

    "Average Fare - Non-Survivors":
        round(not_survived_fare, 2)
}

insights_df = pd.DataFrame(
    list(insights.items()),
    columns=["Insight", "Value"]
)

insights_df.to_csv(
    "key_insights.csv",
    index=False
)

print("\n" + "=" * 70)
print("EDA PROJECT SUMMARY")
print("=" * 70)

print("\nDataset:", "Titanic")

print("Rows:", df.shape[0])

print("Columns:", df.shape[1])

print("\nAnalysis completed:")
print("✓ Statistical Summary")
print("✓ Missing Value Analysis")
print("✓ Duplicate Analysis")
print("✓ Survival Analysis")
print("✓ Gender Analysis")
print("✓ Passenger Class Analysis")
print("✓ Age Analysis")
print("✓ Fare Analysis")
print("✓ Family Size Analysis")
print("✓ Correlation Analysis")
print("✓ Grouped Analysis")
print("✓ Key Insights")

print("\nReports generated:")
print("1. missing_value_report.csv")
print("2. correlation_matrix.csv")
print("3. grouped_analysis_report.csv")
print("4. key_insights.csv")

print("\nVisualizations generated:")
print("1. 01_survival_distribution.png")
print("2. 02_gender_distribution.png")
print("3. 03_survival_by_gender.png")
print("4. 04_passenger_class_distribution.png")
print("5. 05_survival_by_class.png")
print("6. 06_age_distribution.png")
print("7. 07_age_vs_survival.png")
print("8. 08_fare_distribution.png")
print("9. 09_fare_vs_survival.png")
print("10. 10_age_vs_fare.png")
print("11. 11_survival_by_family_size.png")
print("12. 12_correlation_heatmap.png")

print("\n" + "=" * 70)
print("EDA PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)