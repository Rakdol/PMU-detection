FROM amd64/python:3.9-slim

RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

WORKDIR /usr/app

RUN pip install -U pip && \
    pip install pandas psycopg2-binary

COPY data_generator.py data_generator.py
COPY pmu.py pmu.py

ENTRYPOINT ["python", "data_generator.py", "--db-host"]
# Change CMD to solve host finding error
CMD ["localhost"]