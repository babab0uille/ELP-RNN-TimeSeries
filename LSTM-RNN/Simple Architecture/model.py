import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

# ----------------- Step 1: Define the feature description -----------------
feature_description = {
    'sample': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

# ----------------- Step 2: Define a Safe Normalization Function -----------------
def safe_normalize(sample):
    sample_mean = tf.reduce_mean(sample)
    sample_std = tf.math.reduce_std(sample)
    sample_std = tf.maximum(sample_std, 1e-7)
    return (sample - sample_mean) / sample_std

# ----------------- Step 3: Define the Parsing Function -----------------
def _parse_function(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    sample_bytes = parsed['sample']
    sample = tf.io.decode_raw(sample_bytes, tf.float16)
    sample = tf.cast(sample, tf.float32)
    sample = tf.reshape(sample, [-1, 1])
    normalized_sample = safe_normalize(sample)
    normalized_sample = tf.where(tf.math.is_finite(normalized_sample),
                                 normalized_sample,
                                 tf.zeros_like(normalized_sample))
    label = parsed['label']
    return normalized_sample, label

# ----------------- Step 4: Define the Windowing Function -----------------
def create_windows(sample, label, window_size=100, shift=50):
    ds = tf.data.Dataset.from_tensor_slices(sample)
    ds = ds.window(window_size, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size))
    ds = ds.map(lambda w: (w, label))
    return ds

# ----------------- Step 5: Create Windowed Training and Validation Datasets -----------------
batch_size = 32
window_size = 100
shift = 50

train_tfrecord = '/home/eerodriguez3/elp/tfrecords_cherrypicked/train_cherrypick.tfrecord'
val_tfrecord = '/home/eerodriguez3/elp/tfrecords_cherrypicked/validate_cherrypick.tfrecord'

raw_train_dataset = tf.data.TFRecordDataset([train_tfrecord], buffer_size=512 * 1024)
raw_val_dataset = tf.data.TFRecordDataset([val_tfrecord], buffer_size=512 * 1024)

parsed_train_dataset = raw_train_dataset.map(_parse_function)
parsed_val_dataset = raw_val_dataset.map(_parse_function)

windowed_train_dataset = parsed_train_dataset.flat_map(
    lambda sample, label: create_windows(sample, label, window_size, shift)
).filter(lambda w, l: tf.reduce_all(tf.math.is_finite(w)))

windowed_val_dataset = parsed_val_dataset.flat_map(
    lambda sample, label: create_windows(sample, label, window_size, shift)
).filter(lambda w, l: tf.reduce_all(tf.math.is_finite(w)))

windowed_train_dataset = windowed_train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
windowed_val_dataset = windowed_val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Calculate steps manually if cardinality is unknown
train_steps = windowed_train_dataset.cardinality().numpy()
val_steps = windowed_val_dataset.cardinality().numpy()

if train_steps == tf.data.INFINITE_CARDINALITY or train_steps == tf.data.UNKNOWN_CARDINALITY:
    train_steps = sum(1 for _ in windowed_train_dataset)
if val_steps == tf.data.INFINITE_CARDINALITY or val_steps == tf.data.UNKNOWN_CARDINALITY:
    val_steps = sum(1 for _ in windowed_val_dataset)

# ----------------- Step 6: Build the RNN Model -----------------
model = Sequential([
    Input(shape=(window_size, 1)),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

# ----------------- Step 7: Compile the Model -----------------
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ----------------- Step 8: Train the Model -----------------
history = model.fit(
    windowed_train_dataset,
    validation_data=windowed_val_dataset,
    epochs=20,
    steps_per_epoch=train_steps,
    validation_steps=val_steps
)
# Save the trained model
model.save('trained_model.h5')

# Optionally, save training history if needed
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

