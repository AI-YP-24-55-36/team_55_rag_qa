import pandas as pd

def read_data():

    rag_dataset = pd.read_csv('full_dataset.csv')

    # Предобработка текстов
    df = rag_dataset.copy()

    # удаление строк с пустыми значениями (None)
    df = df.dropna()

    # приведем к нижнему регистру тексты во всех колонках
    df['context'] = df['context'].apply(lambda x: x.lower())
    df['question'] = df['question'].apply(lambda x: x.lower())
    df['answer'] = df['answer'].apply(lambda x: x.lower())

    # удаление/замена на пробел знака перевода строки в колонке 'context'
    df['context'] = df['context'].apply(lambda x: x.replace('\n', ' '))
    # удаление найденных дубликатов по 'question'
    df.duplicated(subset=['question'], keep=False)
    df.drop_duplicates(subset = ['question'], keep = 'first', inplace = True)

    # удаление двух строк на другом языке
    df.drop(index=[7453, 10225], inplace=True)
    df.reset_index(drop=True, inplace=True) # обновление индексов

    data  = [
            {"id": i, "context": df.context.tolist()[i],
             "question": df.question.tolist()[i]}
            for i in range(len(df))
            ]

    test_query = df.question.sample(50, random_state=42).tolist()
    ground_truth = df.question.sample(50, random_state=42).index.tolist()

    return data, test_query, ground_truth

