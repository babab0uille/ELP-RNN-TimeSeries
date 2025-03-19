def load_dataset(file_path, batch_size=32):
    import tensorflow as tf
    # Load dataset assuming it's a TFRecord or similar
    dataset = tf.data.TFRecordDataset(file_path)
    # Example parsing function (adjust according to your data structure)
    def _parse_function(example_proto):
        # Define your feature description dictionary
        feature_description = {
            'feature1': tf.io.FixedLenFeature([], tf.float32),
            'feature2': tf.io.FixedLenFeature([], tf.float32),
        }
        return tf.io.parse_single_example(example_proto, feature_description)

    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size)
    return dataset
