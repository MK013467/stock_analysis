FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt --verbose

EXPOSE 5000

ENV NAME Stocks

# Run app.py when the container launches
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
