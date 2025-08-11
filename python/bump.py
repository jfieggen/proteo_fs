import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define the dataset based on user input
columns = ["Feature", "Column 1", "Column 2", "Column 3", "Column 4"]
data = [
    ["SLAMF7", 1, 6, 4, 4],
    ["SLAMF1", 2, 5, 3, 6],
    ["BCMA", 3, 17, 6, 8],
    ["CNTN5", 4, 7, 5, 5],
    ["QPCT", 5, 4, 2, 2],
    ["BAFF", 6, 2, 1, 3],
    ["FCRL5", 7, 12, 8, None],
    ["IL5RA", 8, 9, 7, 14],
    ["LY9", 9, 13, 9, 7],
    ["FRZB", 10, 14, 10, 13],
    ["CD79B", 11, 18, None, None],
    ["ICAM3", 12, 8, None, 17],
    ["APRIL", 13, 1, None, 9],
    ["CD48", 14, 19, None, None],
    ["SMOC1", 15, 10, None, None],
    ["COL18A1", 16, 3, None, None],
    ["HGF", 17, 15, None, None],
    ["EDA2R", 18, 11, None, None],
    ["RNASET2", 19, 16, None, None],
    ["TACI", None, None, None, 1],
    ["FCRLB", None, None, None, 10],
]

# Create a DataFrame
df = pd.DataFrame(data, columns=columns)

# Convert to long format for ranking
df_long = df.melt(id_vars=["Feature"], var_name="Column", value_name="Rank")
df_long.dropna(inplace=True)

# Pivot the data for plotting
df_pivot = df_long.pivot(index="Feature", columns="Column", values="Rank")

# Ensure the correct column order (1 → 2 → 3 → 4)
df_pivot = df_pivot.rename(columns={"Column 1": "1", "Column 2": "2", "Column 3": "3", "Column 4": "4"})
df_pivot = df_pivot[["1", "2", "3", "4"]]  # Explicitly reordering
df_pivot.columns = df_pivot.columns.astype(int)  # Convert to integer for sorting
df_pivot = df_pivot.sort_index(axis=1)  # Sort numerically

# Plot the bump chart
plt.figure(figsize=(14, 8))

# Assign unique colors for each feature
colors = plt.cm.get_cmap("tab20", len(df_pivot))

for i, feature in enumerate(df_pivot.index):
    values = df_pivot.loc[feature].dropna()
    plt.plot(values.index, values, marker="o", label=feature, linestyle="-", color=colors(i))
    
    # Shift text slightly to the left for better spacing
    plt.text(values.index[0] - 0.2, values.iloc[0], feature, ha='right', va='center', fontsize=10, fontweight='bold')

# Formatting the plot
plt.gca().invert_yaxis()  # Lower ranks should be at the top
plt.xticks(df_pivot.columns, labels=["Univariable", "Multivariable (20)", "Multivariable (10)", "XGBoost"])
plt.xlabel("Model Type")
plt.ylabel("Rank")

# Ensure y-axis increments in steps of 1 unit
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.title("Ranking Changes of Features Across Models")
plt.grid(True, linestyle="--", alpha=0.6)

# Show the plot
plt.show()
