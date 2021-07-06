import pandas as pd
import tensorflow as tf
from tensorflow import keras as keras
from keras.layers.experimental.preprocessing import IntegerLookup, StringLookup, Normalization

df = pd.read_csv('Bike_Data.csv')

val_df = df.sample(frac = 0.1, random_state=3)
train_df = df.drop(val_df.index)
print(f"Using {len(train_df)} samples fr training and {len(val_df)} for validation ")

def dataset_from_dataframe(dataframe):
  df = dataframe.copy()
  df.pop("bike_name")
  labels = df.pop("price")

  return tf.data.Dataset.from_tensor_slices((dict(df), labels))

train_ds = dataset_from_dataframe(train_df)
val_ds = dataset_from_dataframe(val_df)
train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature


inps = {

    "city": "string",
    "kms_driven": "float64",
    "owner": "string",
    "age": "float64",
    "power": "float64",
    "brand": "string",


}

def more_preprocess (dataset):
  input_dict = {}
  encoded_dict = {}
  for inp_name, dtype in inps.items():
    print(f"at {inp_name}")
    input_dict[inp_name] = tf.keras.Input(shape=(1,), name=inp_name, dtype=dtype)
    if (dtype == "int64" or dtype== "float64"):
      encoded_dict[inp_name] = encode_numerical_feature(input_dict[inp_name], inp_name, dataset)
    else:
      encoded_dict[inp_name] = encode_categorical_feature(input_dict[inp_name], inp_name, dataset, True)
  return input_dict, encoded_dict



train_input_dict, train_encoded_dict = more_preprocess(train_ds)

input_columns = train_encoded_dict.keys()
train_encoded_list = [train_encoded_dict[x] for x in input_columns]
train_inp_list = [train_input_dict[x] for x in input_columns]

all_encoded_inputs = keras.layers.concatenate(train_encoded_list)


x = keras.layers.Dense(32, activation="relu")(all_encoded_inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(16, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(16, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(1, activation="relu")(x)

model = keras.Model(train_inp_list, x)

model.compile('adam', loss=[keras.losses.mean_squared_error], metrics = [keras.metrics.MeanAbsolutePercentageError()])

callbacks  = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)]

model.fit(train_ds, epochs = 500, validation_data = val_ds, callbacks = callbacks)

print(model.summary())

keras.models.save_model(model, './used_bikes')
