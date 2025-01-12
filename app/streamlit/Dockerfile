FROM python:3.12-slim


WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords
COPY . .

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]

CMD [ "st_app.py" ]