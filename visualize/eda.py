import pandas as pd
from visualize.visualize import (
    plot_correlation_heatmap, 
    plot_G3_correlation, 
    plot_G3_distribution,
    plot_G3_by_categorical_features
)

df = pd.read_csv("data/student-mat.csv")
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

plot_correlation_heatmap(df)
plot_G3_correlation(df)
plot_G3_distribution(df)
plot_G3_by_categorical_features(df, categorical_columns)
