import io
import os
from pathlib import Path

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
data_dir = Path("partitioned_data") / "small-data"
image_name = "17139.jpg"
image_path = data_dir / image_name

# Loads the image into memory
with io.open(image_path, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

# Performs label detection on the image file
response = client.label_detection(image=image)
labels = response.label_annotations

print('Labels:')
for label in labels:
    print(label.description)
