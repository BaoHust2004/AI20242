import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

def save_fig(fig, filename):
    os.makedirs("visuals", exist_ok=True)
    fig.savefig(f"visuals/{filename}", bbox_inches='tight')
    plt.close(fig)

def plot_mae_comparison(mae_scores):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=list(mae_scores.keys()), y=list(mae_scores.values()), palette="viridis", ax=ax)
    ax.set_title("MAE Comparison Between Models")
    ax.set_ylabel("MAE")
    for i, v in enumerate(mae_scores.values()):
        ax.text(i, v + 0.1, f"{v:.2f}", ha='center')
    save_fig(fig, "model_comparison_MAE.png")

def plot_feature_importance(model, feature_names, model_name):
    importances = model.feature_importances_
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df = df.sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Importance", y="Feature", palette="Blues_r", ax=ax)
    ax.set_title(f"Feature Importance - {model_name}")
    save_fig(fig, f"feature_importance_{model_name}.png")

def plot_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    save_fig(fig, "correlation_heatmap.png")

def plot_G3_correlation(df):
    corr_with_G3 = df.corr(numeric_only=True)["G3"].drop("G3").sort_values(key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=corr_with_G3.values, y=corr_with_G3.index, palette="mako", ax=ax)
    ax.set_title("Correlation of Features with G3")
    save_fig(fig, "G3_correlation_barplot.png")

def plot_G3_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(df["G3"], kde=True, color="purple", ax=ax)
    ax.set_title("Distribution of G3")
    save_fig(fig, "G3_distribution.png")

def plot_G3_by_categorical_features(df, categorical_cols):
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x=col, y="G3", palette="Set2", ax=ax)
        ax.set_title(f"G3 by {col}")
        save_fig(fig, f"G3_by_{col}.png")
