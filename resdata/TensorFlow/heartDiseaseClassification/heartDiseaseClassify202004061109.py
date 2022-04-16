# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('heart.csv')
print(df.head())

train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = df.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices(
        (dict(dataframe), labels)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size).repeat()
# tf.print(list(train_ds))
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
    tf.print(list(feature_batch.keys()))
    tf.print(list(feature_batch['age']))
    tf.print(list(label_batch))

example_ds = next(iter(train_ds))
example_batch = example_ds[0]
example_label = example_ds[1]


def demo(feature_column):
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())
    # tf.print(feature_layer(example_batch))


age = tf.feature_column.numeric_column('age')
demo(age)

age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)

thal = tf.feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible']
)
thal_one_hot = tf.feature_column.indicator_column(thal)
demo(thal_one_hot)

# embedding_column的输入是categorical column
thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)

thal_hashed = tf.feature_column.categorical_column_with_hash_bucket(
    'thal', hash_bucket_size=20
)
demo(tf.feature_column.indicator_column(thal_hashed))

crossed_feature = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=16)
demo(tf.feature_column.indicator_column(crossed_feature))

# Select columns using for features
feature_columns = []

# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(tf.feature_column.numeric_column(header))

# bucketized cols
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35,
                                                                   40, 45, 50, 55,
                                                                   60, 65])
feature_columns.append(age_buckets)

# indicator cols
thal = tf.feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible']
)
thal_one_hot = tf.feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = tf.feature_column.crossed_column(
    [age_buckets, thal], hash_bucket_size=1000
)
crossed_feature = tf.feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential(
    [
        feature_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=10, steps_per_epoch=10)

loss, accuracy = model.evaluate(test_ds)
print(loss, accuracy)
