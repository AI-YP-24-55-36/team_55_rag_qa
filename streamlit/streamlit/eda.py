
import matplotlib.pyplot as plt
import seaborn as sns


def length (df):
    columns = df.columns
    for col in columns:
        df[f'length_{col}'] = df[col].astype('str').apply(lambda text: len(text.split()))
    return df

def plot_length(df, col):
    sns.set_theme(palette='pastel', font_scale=0.9)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))
    sns.histplot(df[col[0]], ax=axes[0]).set_title(f"Количество слов в колонке {col[0]}")
    sns.histplot(df[col[1]], ax=axes[1]).set_title(f"Количество слов в колонке {col[1]}")
    sns.histplot(df[col[2]], ax=axes[2]).set_title(f"Количество слов в колонке {col[2]}")
    plt.subplots_adjust(hspace=0.5)
    return fig, axes




