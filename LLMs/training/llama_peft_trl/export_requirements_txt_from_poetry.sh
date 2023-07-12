#!/bin/bash

# check if requrirements.txt exists
if [ -f requirements.txt ]; then
    echo "requirements.txt already exists. Remove it first."
    rm requirements.txt
fi

poetry export --without-hashes --format=requirements.txt > requirements.txt