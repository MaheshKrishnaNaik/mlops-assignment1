# %% [markdown]
# # EDA - Heart Disease UCI
# Run: `python notebooks/eda.py` (VS Code / Jupytext treat `# %%` as cells). Plots go to `screenshots/eda/`.

# %%
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402
from src.data import load_clean  # noqa: E402

OUT = config.ROOT / "screenshots" / "eda"
Path(OUT).mkdir(parents=True, exist_ok=True)
df = load_clean()
print(df.shape)
print(df.describe().T)

# %% Class balance
ax = sns.countplot(x=config.TARGET, data=df)
ax.set_title("Class balance (0 = no disease, 1 = disease)")
plt.savefig(OUT / "class_balance.png", dpi=120, bbox_inches="tight")
plt.close()
print(df[config.TARGET].value_counts(normalize=True))

# %% Histograms
df[config.NUMERIC].hist(bins=20, figsize=(12, 7))
plt.suptitle("Numeric feature distributions")
plt.savefig(OUT / "histograms.png", dpi=120, bbox_inches="tight")
plt.close()

# %% Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap")
plt.savefig(OUT / "correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()

# %% Features vs target
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["cp", "thal", "ca"]):
    sns.countplot(x=col, hue=config.TARGET, data=df, ax=ax)
plt.savefig(OUT / "categorical_vs_target.png", dpi=120, bbox_inches="tight")
plt.close()
print("EDA plots saved to", OUT)
