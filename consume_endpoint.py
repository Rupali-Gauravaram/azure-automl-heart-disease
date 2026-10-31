"""Call an Azure ML managed online endpoint (key auth).

Set these before running (never hard-code the key):
    export AZURE_ML_ENDPOINT="https://<endpoint-name>.<region>.inference.ml.azure.com/score"
    export AZURE_ML_KEY="<primary key>"
"""
import json
import os
import urllib.request

# Column order must match the model's input schema (see MLmodel in the zip)
columns = []   # e.g. ["age", "sex", "bmi", ...]
values = []    # one row of values in the same order

body = json.dumps({"input_data": {"columns": columns, "data": [values]}}).encode("utf-8")
headers = {"Content-Type": "application/json",
           "Authorization": f"Bearer {os.environ['AZURE_ML_KEY']}"}
req = urllib.request.Request(os.environ["AZURE_ML_ENDPOINT"], body, headers)

try:
    print(urllib.request.urlopen(req).read().decode("utf-8"))
except urllib.error.HTTPError as error:
    print("Error:", error.code, error.read().decode("utf-8"))
