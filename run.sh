#!/bin/bash

set -eu

PORT=${PORT:-8501}
APP_NAME=${APP_NAME:-"main.py"}

streamlit run ${APP_NAME} \
    --server.port ${PORT}