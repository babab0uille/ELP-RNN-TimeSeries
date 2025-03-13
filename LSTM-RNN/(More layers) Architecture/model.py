import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import cohen_kappa_score
import pickle

# ----------------- Define Feature Description -----------------
feature_description = {
    'sample': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

# ----------------- Safe Normalization Function -----------------
def safe_normalize(sample):
    sample_mean = tf.reduce_mean(sample)
    sample_std = tf.math.reduce_std(sample)
    sample_std = tf.maximum(sample_std, 1e-7)
    return (sample - sample_mean) / sample_std

# ----------------- Parsing Function -----------------
def _parse_function(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    sample = tf.io.decode_raw(parsed['sample'], tf.float16)
    sample = tf.cast(sample, tf.float32)
    sample = tf.reshape(sample, [-1, 1])
    sample = safe_normalize(sample)
    sample = tf.where(tf.math.is_finite(sample), sample, tf.zeros_like(sample))
    label = tf.cast(parsed['label'], tf.float32)
    return sample, label

# ----------------- Windowing Function -----------------
def create_windows(sample, label, window_size=100, shift=50):
    ds = tf.data.Dataset.from_tensor_slices(sample)
    ds = ds.window(window_size, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size))
    ds = ds.map(lambda w: (w, label))
    return ds

# ----------------- Custom F1 Score -----------------
def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    return 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

# ----------------- Prepare Datasets -----------------
batch_size = 32
train_tfrecord = '/home/eerodriguez3/elp/tfrecords_cherrypicked/train_cherrypick.tfrecord'
val_tfrecord = '/home/eerodriguez3/elp/tfrecords_cherrypicked/validate_cherrypick.tfrecord'

raw_train_dataset = tf.data.TFRecordDataset([train_tfrecord]).cache().shuffle(1000).map(_parse_function)
raw_val_dataset = tf.data.TFRecordDataset([val_tfrecord]).cache().map(_parse_function)

windowed_train_dataset = raw_train_dataset.flat_map(lambda s, l: create_windows(s, l)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
windowed_val_dataset = raw_val_dataset.flat_map(lambda s, l: create_windows(s, l)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ----------------- Build Enhanced RNN Model -----------------
model = Sequential([
    Input(shape=(100, 1)),
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(32),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# ----------------- Compile Model with Advanced Metrics -----------------
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score])

# ----------------- Training Callbacks -----------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# ----------------- Train the Model -----------------
history = model.fit(
    windowed_train_dataset,
    validation_data=windowed_val_dataset,
    epochs=3,
    steps_per_epoch=130,
    callbacks=[early_stop, lr_scheduler]
)

# ----------------- Save Model and Training History -----------------
model.save('enhanced_trained_model.h5')
with open('enhanced_training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# ----------------- Calculate Cohen's Kappa -----------------
val_preds = model.predict(windowed_val_dataset)
val_labels = []
for _, labels in windowed_val_dataset:
    val_labels.extend(labels.numpy())
val_preds_binary = (val_preds.flatten() > 0.5).astype(int)
kappa = cohen_kappa_score(val_labels, val_preds_binary)
print(f"Cohen's Kappa: {kappa}")

