FROM amd64/python:3.9-slim

RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

WORKDIR /usr/app

RUN pip install -U pip && \
    pip install pandas psycopg2-binary

COPY data_generator.py data_generator.py
COPY pmu.csv pmu.csv
COPY ./docker-entrypoint.sh docker-entrypoint.sh

RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["/usr/app/docker-entrypoint.sh"]

CMD ["--db-host", "localhost", "--csv-path", "/usr/app/pmu.csv"]