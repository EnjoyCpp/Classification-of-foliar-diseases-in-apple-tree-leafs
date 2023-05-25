import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt 
import glob as gb
import efficientnet.tfkeras as efn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import load_img
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from IPython.display import SVG, Image
import cv2
from sklearn.preprocessing import MultiLabelBinarizer
import os
import hashlib

from PIL import Image
import albumentations as A

print(tf.__version__)

# tf.config.set_visible_devices([], 'GPU')

gpus = tf.config.list_physical_devices('GPU'); print(gpus)
if len(gpus)==1: strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
else: strategy = tf.distribute.MirroredStrategy()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
print('Mixed precision enabled')

def load_image(filename):
    image = load_img("./data/img_sz_256/" + filename)
    plt.imshow(image) 

def load_random_image(filenames):
    sample = random.choice(filenames)
    image = load_img("./data/img_sz_256/" + sample)
    plt.imshow(image)   

def load_image_for_augmentation(image_path):
    image = cv2.imread(image_path)
    if image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    image = np.array(image)
    return image          

# adding path we will use in this nothebook 
def format_resized_image_path_gcs(st):
    return GCS_DS_PATH + '/img_sz_256/' + st
def format_tpu_path(st):
    return './data/img_sz_256/' + st

train_df = pd.read_csv("./train.csv")

RESIZED_IMAGE_PATH = "./data/img_sz_256/"

train_df.labels.value_counts()

initial_length = len(train_df)

# create dictionary to store hashes and paths
hashes = {}

duplicates = []
originals = []

# loop over rows in the dataframe
for index, row in train_df.iterrows():
    # get the filename of the image
    filename = row['image']
    
    # compute the hash of the image
    with open(os.path.join(RESIZED_IMAGE_PATH, filename), 'rb') as f:
        hash = hashlib.md5(f.read()).hexdigest()
    
    # check if hash already exists in dictionary
    if hash in hashes:
        duplicates.append(filename)
        
        originals.append(hashes[hash])
        # delete duplicate row from dataframe
        train_df.drop(index, inplace=True)
    else:
        # add hash and path to dictionary
        hashes[hash] = filename

# print number of duplicates found
print(f"Number of duplicates found: {initial_length-len(hashes)}")

# Extract the labels column from the dataframe and store it in a list
labels = train_df['labels'].tolist()

# Create a set of unique labels
unique_labels = set()

# Iterate through each label in the list of labels and add it to the set of unique labels
for label in labels:
    unique_labels.update(label.split())

# Print the number of unique labels
print(unique_labels, "suma:", len(unique_labels))

# Select the six most common unique labels
common_labels = [label[0] for label in pd.Series(labels).str.split(expand=True).stack().value_counts()[:6].items()]
# Create a MultiLabelBinarizer object with the six common labels
mlb = MultiLabelBinarizer(classes=common_labels)

# Transform the labels into a binary matrix with one-hot encoding for the six common labels
label_matrix = mlb.fit_transform(train_df['labels'].str.split())

# Create a new dataframe with the one-hot encoded labels
label_df = pd.DataFrame(label_matrix, columns=common_labels)

# fixes train_df row number error
train_df.reset_index(drop=True, inplace=True)
label_df.reset_index(drop=True, inplace=True)

# Concatenate the new label dataframe with the original dataframe
new_df = pd.concat([train_df, label_df], axis=1)

# Drop the original labels column from the new dataframe
train_df = new_df.drop('labels', axis=1)
train_df.head()



# Training and validation split

#===============================================================
#===============================================================
# X = train_df.image.apply(format_resized_image_path_gcs).values
X = train_df.image.apply(format_tpu_path).values
y = np.float32(train_df.loc[:, 'healthy':'scab'].values)

df = train_df

# Get the hot encoded label columns
label_cols = [col for col in df.columns if not col.startswith('image')]

# Split the dataframe into features (X) and labels (y)
# X = df.drop(label_cols, axis=1)
y = df[label_cols]

# Determine the size of the validation set
validation_size = 0.2

# Split the data into training and validation sets while ensuring that the label distribution is balanced in both sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, stratify=y, random_state=25)

arr_df = pd.DataFrame(X_train, columns=['image'])

# Reset the index of arr_df
arr_df = arr_df.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

train_df = pd.concat([arr_df, y_train], axis=1)

def cutout_image(image):
    image_array = np.array(image)
    augmented_array = A.CoarseDropout(max_holes=8, max_height=12, max_width=12, always_apply=True)(image=image_array)['image']

    return Image.fromarray(augmented_array)

def elastic_transform_image(image):
    image_array = np.array(image)
    augmented_array = A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, always_apply=True)(image=image_array)['image']

    return Image.fromarray(augmented_array)

def gaussian_noise_image(image):
    image_array = np.array(image)
    augmented_array = A.GaussNoise(var_limit=(15.0, 60.0), always_apply=True)(image=image_array)['image']

    return Image.fromarray(augmented_array)

df = train_df

labels = unique_labels

# Get the counts of each label
label_counts = df[list(unique_labels)].sum()

# Calculate the number of rows to be added for each label
total_rows_added = int(len(df) * 0.03)
rows_added_per_label = (total_rows_added * label_counts / label_counts.sum()).astype(int)

# Find the maximum value
max_value = rows_added_per_label.max()

# Make the smallest value the largest
rows_added_per_label = max_value - rows_added_per_label + max_value

print(rows_added_per_label)

# Create a new DataFrame for the augmented images
cutout_df = pd.DataFrame(columns=df.columns)
elastic_transform_df = pd.DataFrame(columns=df.columns)
gaus_df = pd.DataFrame(columns=df.columns)


# Create a new directory to save augmented images
cutout_dir = 'cutout_images'
os.makedirs(cutout_dir, exist_ok=True)

# Create a new directory to save augmented images
elastic_dir = 'elastic_images'
os.makedirs(elastic_dir, exist_ok=True)

# Create a new directory to save augmented images
gaus_dir = 'gaus_images'
os.makedirs(gaus_dir, exist_ok=True)

# Track the selected indices
selected_indices_set = set()

# Augment the data for each label
for label in labels:
    # Get the indices of rows with the current label
    label_indices = df[df[label] == 1].index
    # Convert label indices to a NumPy array
    label_indices = np.array(label_indices)

    # Shuffle the label indices randomly
    np.random.shuffle(label_indices)
    
    # Get the number of rows to be added for the current label
    rows_to_add = rows_added_per_label[label]

    # Select the indices for augmentation, excluding the ones already selected
    selected_indices = np.random.choice(
    np.setdiff1d(label_indices, list(selected_indices_set)),
    size=min(rows_to_add, len(label_indices)),
    replace=False)

    # Update the set of selected indices
    selected_indices_set.update(selected_indices)

    # Augment the selected rows and update the image paths in the dataframe
    for index in selected_indices:
        image_path = df.at[index, 'image']
        
        image = load_image_for_augmentation(image_path)
        
        # Apply cutout
        augmented_image = cutout_image(image)
        
        # Save the augmented image
        augmented_image_path = os.path.join(cutout_dir, f'cutout_{label}_{index}.png')
        # Save the augmented image
        augmented_image.save(augmented_image_path)

        new_row = df.iloc[index].copy()
        new_row['image'] = augmented_image_path

        # save to cutout dataframe
        cutout_df.loc[len(cutout_df)] = new_row.values
        
        
        # Apply elastic transformations
        augmented_image = elastic_transform_image(image)
        
        # Save the augmented image
        augmented_image_path = os.path.join(elastic_dir, f'elastic_{label}_{index}.png')
        augmented_image.save(augmented_image_path)

        new_row = df.iloc[index].copy()
        new_row['image'] = augmented_image_path
        
        # save to elastic dataframe
        elastic_transform_df.loc[len(elastic_transform_df)] = new_row.values
        
        
        # Apply gaussian noise
        augmented_image = gaussian_noise_image(image)
        
        # Save the augmented image
        augmented_image_path = os.path.join(gaus_dir, f'gaus_{label}_{index}.png')
        augmented_image.save(augmented_image_path)

        new_row = df.iloc[index].copy()
        new_row['image'] = augmented_image_path
        
        # save to gaus dataframe
        gaus_df.loc[len(gaus_df)] = new_row.values

# Save the cutout DataFrame to a new CSV file
augmented_dataframe_path = 'cutout_df.csv'
cutout_df.to_csv(augmented_dataframe_path, index=False)

# Save the elastic DataFrame to a new CSV file
augmented_dataframe_path = 'elastic_transform_df.csv'
elastic_transform_df.to_csv(augmented_dataframe_path, index=False)


# Save the elastic DataFrame to a new CSV file
augmented_dataframe_path = 'gaus_df.csv'
gaus_df.to_csv(augmented_dataframe_path, index=False)

# Reset the index 
train_df.reset_index(drop=True, inplace=True)
cutout_df.reset_index(drop=True, inplace=True)
elastic_transform_df.reset_index(drop=True, inplace=True)

# Concatenate the two dataframes vertically
combined_df = pd.concat([train_df, cutout_df, elastic_transform_df, gaus_df], axis=0)

# Reset the index of the combined dataframe
combined_df.reset_index(drop=True, inplace=True)

print("Original dataframe length:", len(train_df),
      "\nCutout dataframe length:", len(cutout_df),
      "\nElastic transform dataframe length:", len(elastic_transform_df),
      "\nGaussian noise dataframe length:", len(gaus_df))

print("------------------------------------------------------------")
print("Combined length:", len(combined_df))
print("____________________________________________________________")
print("Added length:", len(combined_df) - len(train_df))

X_train = combined_df[['image']]
y_train = combined_df.drop('image', axis=1)
y_train = y_train.to_numpy()
y_train = y_train.astype(np.float32)

# Convert DataFrame column to NumPy array and change shape
X_train = X_train['image'].values.squeeze()

import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

if(strategy.num_replicas_in_sync != 1 ):
    print('TPU used')
    BATCH_SIZE = 48 * strategy.num_replicas_in_sync
else:
    BATCH_SIZE = 1
    
print("Batch size:", BATCH_SIZE)

STEPS_PER_EPOCH = y_train.shape[0] // BATCH_SIZE
VALIDATION_STEPS = y_val.shape[0] // BATCH_SIZE

print ("Steps per epoch: ", STEPS_PER_EPOCH, "\nValidation steps: ", VALIDATION_STEPS)

image_height = 380
image_width = 380


def decode_image(filename, label=None, image_size=(image_height, image_width)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label

def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_hue(image, max_delta=0.3)
    
    if label is None:
        return image
    else:
        return image, label

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .map(decode_image, num_parallel_calls = AUTO)
    .map(data_augment, num_parallel_calls = AUTO)
    .repeat()
    .shuffle(256)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_val, y_val))
    .map(decode_image, num_parallel_calls = AUTO)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

tensorboard = TensorBoard(log_dir = 'logs')

checkpoint = ModelCheckpoint("effnet.h5", monitor="val_accuracy",save_best_only=True,
                             mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.1, patience = 2, min_delta = 0.001,
                              mode='auto', verbose=1, min_lr=1e-9)

# Choosing to monitor val_accuracy since it is classification task and accuracy is more important
early_stop=EarlyStopping(monitor='val_accuracy', restore_best_weights= True,
                             patience=7, verbose=1)

filename='history.csv'
history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)


callback_options = [tensorboard, checkpoint, reduce_lr, early_stop, history_logger]

effnet = Sequential([efn.EfficientNetB4(input_shape=(image_height, image_width,3), weights='noisy-student', include_top=False)])
# effnet.trainable = False

model = effnet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dense(256,activation='relu')(model)
model = tf.keras.layers.Dense(128,activation='relu')(model)
model = tf.keras.layers.BatchNormalization()(model)
model = tf.keras.layers.Dropout(rate=0.2)(model)
model = tf.keras.layers.Dense(64,activation='relu')(model)
model = tf.keras.layers.BatchNormalization()(model)
model = tf.keras.layers.Dropout(rate=0.2)(model)
model = tf.keras.layers.Dense(6,activation='sigmoid')(model)
model = tf.keras.models.Model(inputs=effnet.input, outputs = model)

model.summary()

import json

# Count the number of occurrences for each label
label_counts = np.sum(y_train, axis=0)

# Calculate the class weights based on label frequencies
total_samples = y_train.shape[0]
class_weights = total_samples / (len(label_counts) * label_counts)

class_weights_dict = dict(zip(range(len(label_counts)), class_weights))

print(class_weights_dict)

model.compile(loss='binary_crossentropy',optimizer = 'Adam', metrics= ['accuracy'])

history = model.fit(train_dataset,
                    epochs=30,
                    callbacks = callback_options,
                    class_weight=class_weights_dict,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset,
                    validation_steps=VALIDATION_STEPS)

history_df = pd.DataFrame(history.history)

# Save the training history as a CSV file
history_df.to_csv('training_history.csv', index=False)

plt.figure()
fig,(ax1, ax2)=plt.subplots(1,2,figsize=(19,7))
ax1.plot(history.history['loss'])
ax1.plot(history.history['val_loss'])
ax1.legend(['training','validation'])
ax1.set_title('loss')
ax1.set_xlabel('epoch')

ax2.plot(history.history['accuracy'])
ax2.plot(history.history['val_accuracy'])
ax2.legend(['training','validation'])
ax2.set_title('Acurracy')
ax2.set_xlabel('epoch')


