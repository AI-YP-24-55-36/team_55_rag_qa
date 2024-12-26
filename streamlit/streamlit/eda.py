import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import defaultdict
from collections import Counter

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# длины текстов в словах(токенах)
def length(df):
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


# частотность
# Зафиксируем список стоп-слов
# функция, токенизирует по словам переданный ей текст в список слов, возвращает корпус токенов
def corpus(text):
    words = text.str.split().values.tolist()
    corpus = [word.lower() for i in words for word in i]
    return corpus

# функция для создания столбчатой диаграммы для 15 наиболее часто встречамых стоп-слов и
def plot_top_words(text):
    stop = set(nltk.corpus.stopwords.words('english') + ['-', '-', '–', '&', '/'])
    sns.set_theme(palette='pastel', font_scale=0.9)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    plt.suptitle(f'Наиболее часто встречающиеся стоп-слова и топ _НЕ_ стоп-слов столбец', fontsize=12)

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

    # plt.bar(z, w, ax=axes[0])

    counter = Counter(corp)
    most = counter.most_common()
    x, y = [], []
    # отбор слов, которых нет в списке стоп-слов
    for word, count in most[:99]:
        if (word not in stop):
            x.append(word)
            y.append(count)
    sns.barplot(x=y, y=x, hue=y, ax=axes[1])

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    return fig, axes


# plot_top_words(df['context'], 'context')

def plot_wordcloud(text):
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
    plt.title(f'Облако слов колонка с текстами\n', fontsize=15)
    plt.show()
    return fig, ax
