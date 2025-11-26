import json
import time

import pandas as pd
from confluent_kafka import Producer

conf = {
    "bootstrap.servers": "localhost:9092"  
}
producer = Producer(conf)
topic = "engine-data"

df = pd.read_csv("dataset/PM_train.csv")
num_features = df.shape[1]

def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed for record: {err}")
    else:
        print(f"Record produced to {msg.topic()} [{msg.partition()}]")


for idx, row in df.iterrows():
    data_point = {f"sensor_{i}": row[i] for i in range(num_features)}
    producer.produce(topic, json.dumps(data_point), callback=delivery_report)
    producer.poll(0)  

    time.sleep(0.1)  

producer.flush()
print("Finished streaming CSV data to Kafka")
