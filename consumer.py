import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow as tf
from confluent_kafka import Consumer, KafkaError
import json
import logging

tf.config.set_visible_devices([], 'GPU')
logging.basicConfig(level=logging.INFO)

conf = {
    'bootstrap.servers': 'localhost:9092',  
    'group.id': 'engine-predictions',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(conf)
topic = "engine-data"
consumer.subscribe([topic])

binary_model = load_model("models/model.keras")
rul_model = load_model("models/model.keras")

sequence_length = 50
num_features = 24

buffer = []  

print("Listening to Kafka topic...")

try:
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                logging.error(msg.error())
            continue

        data_point = json.loads(msg.value().decode('utf-8'))
        features = [data_point[f"sensor_{i}"] for i in range(num_features)]
        buffer.append(features)

        if len(buffer) >= sequence_length:
            sequence = np.array(buffer[-sequence_length:]).reshape(1, sequence_length, num_features)

            prediction = binary_model.predict(sequence)[0][0]
            status = "NOT damaged" if round(prediction) else "DAMAGED"

            rul_pred = rul_model.predict(sequence)[0][0]
            cycles_left = round(1 / rul_pred)

            print(f"Engine status: {status}, Cycles left: {cycles_left}")

except KeyboardInterrupt:
    print("Stopping consumer...")
finally:
    consumer.close()
