FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV NAME Stocks

# Run app.py when the container launches
CMD ["python3", "app.py"]