import re
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd


def length(df):
    """Функция расчет длин текстов в словах (токенах)"""
    columns = df.columns
    for col in columns:
        df[f'length_{col}'] = df[col].astype('str').apply(lambda text: len(text.split()))
    return df


def plot_length(df, col):
    """Функция построения графиков слов"""
    fig = make_subplots(rows=3, cols=1, subplot_titles=(f"{col[0]}", f"{col[1]}", f"{col[2]}"))
    fig1 = px.histogram(df, x=col[0])
    fig2 = px.histogram(df, x=col[1])
    fig3 = px.histogram(df, x=col[2])
    fig.add_trace(fig1['data'][0], row=1, col=1)
    fig.add_trace(fig2['data'][0], row=2, col=1)
    fig.add_trace(fig3['data'][0], row=3, col=1)
    fig.update_layout(
        title="Распределение количества слов по каждой колонке с текстом",
        xaxis_title=f"{col[0]}",
        yaxis_title="Количество",
        height=1000,
        width=800,
        template="seaborn",
    )
    fig.update_traces(marker_color='orange', opacity=0.5)
    return fig


def corpus(text):
    """Функция токенизирует по словам переданный ей текст в список слов, возвращает корпус токенов"""
    words = text.str.split().values.tolist()
    corpus_ = [word.lower() for i in words for word in i]
    return corpus_


def plot_top_words(text):
    """Функция для создания столбчатой диаграммы для 15 наиболее часто встречамых стоп-слов"""
    stop = set(nltk.corpus.stopwords.words('english') + ['-', '-', '–', '&', '/'])
    sns.set_theme(palette='pastel', font_scale=0.9)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    plt.suptitle("Топ частотных стоп-слов и топ _НЕ_ стоп-слов", fontsize=12)

    # токенизация на слова
    corp = corpus(text)
    # создание словаря с слово : количество вхождений
    dic = defaultdict(int)
    for word in corp:
        if word in stop:
            dic[word.lower()] += 1
        #   сортировка словаря
    top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:15]
    z, w = zip(*top)
    sns.barplot(x=w, y=z, ax=axes[0])

    counter = Counter(corp)
    most = counter.most_common()
    x, y = [], []
    # отбор слов, которых нет в списке стоп-слов
    for word, count in most[:99]:
        if word not in stop:
            x.append(word)
            y.append(count)
    sns.barplot(x=y, y=x, hue=y, ax=axes[1])

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    return fig, axes


def plot_wordcloud(text):
    """Функция создания облака слов"""
    # соединяем весь текст в список для подачи в wordcloud.generate
    STOPWORDS = set(nltk.corpus.stopwords.words('english') + ['-', '-', '–', '&', '/'])

    text = ' '.join(text.to_list())
    wordcloud = WordCloud(
        background_color='white',
        stopwords=STOPWORDS,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)
    wordcloud = wordcloud.generate(str(text))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    ax.imshow(wordcloud)
    ax.axis('off')
    plt.title("Облако слов колонка с текстами\n", fontsize=15)
    plt.show()
    return fig, ax


def words_only(text):
    """Функция для оставления только слов"""
    return " ".join(re.compile("[A-Za-z]+").findall(text))


def remove_word(text):
    """Функция удаления слов меньше 2-х букв"""
    return " ".join([token for token in text.split() if len(token) > 3])


def remove_stopwords(text):
    """Функция удаление стоп слов"""
    mystopwords = set(nltk.corpus.stopwords.words('english') + ['-', '-', '–', '&', '/'])
    try:
        return " ".join([token for token in text.split() if token not in mystopwords])
    except Exception:
        return ""


def prep(text):
    """Функция препроцессинга текстов"""
    return remove_stopwords(remove_word(words_only(text.lower())))


def plot_tsne(text):
    """Функция построения tsne"""
    corpus_ = text.apply(lambda x: prep(x))
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(corpus_)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X.toarray())
    feature_names = vectorizer.get_feature_names_out()
    word_dict = {i: feature_names[i] for i in range(len(feature_names))}

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1])

    for i, word in word_dict.items():
        ax.annotate(word, (X_tsne[i, 0], X_tsne[i, 1]))
    plt.title('t-SNE Визуализация для 100 самых частых слов', fontsize=15)
    plt.show()
    return fig, ax


def plot_bench(times):
    """Функция получения бенчмарков"""
    sns.set_theme(palette='pastel', font_scale=0.9)
    times_s = pd.Series(times)
    fig, ax = plt.subplots()
    sns.lineplot(x=times_s.index, y=times_s.values, ax=ax, color="red")
    ax.set_title("Время извлечения ответов из базы (на 100 сэмплах)")
    ax.set_xlabel('вопрос')
    ax.set_ylabel('время')
    return fig, ax
