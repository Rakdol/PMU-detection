#!/bin/sh
echo "Executed command: python data_generator.py $@" > /usr/app/command.log
exec python data_generator.py "$@"