FROM python:3.10

ENV PROJECT_DIR=pmu_detection

WORKDIR /${PROJECT_DIR}
ADD ./requirements.txt /${PROJECT_DIR}/
RUN apt-get -y update && \
    apt-get -y install apt-utils gcc curl && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

COPY ./images/ /${PROJECT_DIR}/images/
COPY ./main.py /${PROJECT_DIR}/main.py 
COPY ./run.sh /${PROJECT_DIR}/run.sh

RUN chmod +x /${PROJECT_DIR}/run.sh

CMD ["./run.sh"]