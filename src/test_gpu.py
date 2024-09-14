"""
Python script to test if a local GPU is available for TensorFlow.
Using a local GPU is suggested to avoid long training times and high costs.

--- Execution ---
Run the following command from the root folder:
python src/test_gpu.py

"""

import tensorflow as tf
print(tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
	try:
		logical_gpus = tf.config.list_logical_devices("GPU")
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		print(e)
else:
	print("NO GPU")