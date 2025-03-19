import tensorflow as tf

# ----------------- Count Samples in TFRecord -----------------
def count_tfrecord_samples(tfrecord_file):
    """
    Counts the number of samples in a TFRecord file.
    """
    return sum(1 for _ in tf.data.TFRecordDataset(tfrecord_file))

num_train_samples = count_tfrecord_samples('/home/eerodriguez3/elp/tfrecords_cherrypicked/train_cherrypick.tfrecord')
num_val_samples = count_tfrecord_samples('/home/eerodriguez3/elp/tfrecords_cherrypicked/validate_cherrypick.tfrecord')

print(f"Training Samples: {num_train_samples}")
print(f"Validation Samples: {num_val_samples}")

# ----------------- Inspect TFRecord Samples -----------------
def inspect_tfrecord(tfrecord_file, num_samples=5):
    """
    Inspects and prints the first few examples from a TFRecord file.
    """
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

    for raw_record in raw_dataset.take(num_samples):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)

# Inspect first few samples from the training TFRecord
print("\nInspecting Training TFRecord:")
inspect_tfrecord('/home/eerodriguez3/elp/tfrecords_cherrypicked/train_cherrypick.tfrecord')

# Inspect first few samples from the validation TFRecord
print("\nInspecting Validation TFRecord:")
inspect_tfrecord('/home/eerodriguez3/elp/tfrecords_cherrypicked/validate_cherrypick.tfrecord')

