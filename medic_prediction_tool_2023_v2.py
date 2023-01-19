import os
import numpy as np
from keras.layers import Input, Dense, Embedding, LSTM, Concatenate
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
#from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
from keras.layers import Input, Dense, Embedding, LSTM, Concatenate
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from keras.layers import Conv2D, MaxPooling2D , Flatten
from tensorflow.keras.layers import Conv2D

max_length=128

# Load the image data from directory
def load_images_from_dir(directory, extension='.jpg'):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            filepath = os.path.join(directory, filename)
            image = load_img(filepath, target_size=(150, 150))
            image = img_to_array(image)
            images.append(image)
    return np.asarray(images)
 
# Load the text data and labels from directory
def load_text_from_dir(directory, extension='.txt'):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                text = f.read()
                texts.append(text)
    return texts
 
def load_labels_from_dir(directory, extension='.txt'):
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                label = int(f.read())
                labels.append(label)
    return labels
 
# Read the data for prediction
def load_data_from_file(filepath, extension='.jpg'):
    if filepath.endswith(extension):
        image = load_img(filepath, target_size=(150, 150))
        image = img_to_array(image)
        return image
    else:
        raise ValueError("Invalid file type")

# Load the training data
image_data = load_images_from_dir('data/train')
numbers_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
texts = load_text_from_dir('data/texts')
labels = load_labels_from_dir('data/labels')
# dumy text#texts="sample ahhdha", "sample two"

# Preprocess the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=max_length)

# Define the inputs
image_input = Input(shape=(150, 150, 3))
numbers_input = Input(shape=(3,))
text_input = Input(shape=(1,))
 
# Define the image network
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
image_output = Dense(64, activation='relu')(x)
 
# Define the numbers network
x = Dense(16, activation='relu')(numbers_input)
x = Dense(8, activation='relu')(x)
numbers_output = Dense(1, activation='sigmoid')(x)
 
# Define the text network
x = Embedding(len(tokenizer.word_index) + 1, 32, input_length=max_length)(text_input)
x = LSTM(64)(x)
text_output = Dense(1, activation='sigmoid')(x)
 
# Concatenate the outputs
merged = Concatenate()([image_output, numbers_output, text_output])
 
# Add additional hidden
 
# Add additional hidden layers
x = Dense(64, activation='relu')(merged)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
 
# Define the final output
final_output = Dense(1, activation='sigmoid')(x)
 
# Create the model
model = Model(inputs=[image_input, numbers_input, text_input], outputs=final_output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 
# Train the model
model.fit([image_data, numbers_data, data], labels, epochs=10)
 
# Visualize the model
plot_model(model, to_file='model.png', show_shapes=True)
 
# Read the data for prediction
filepath = 'data/prediction/image.jpg'
image_data = load_data_from_file(filepath)
numbers_data = np.array([[1, 2, 3]])
text = "This is a sample text"

# Preprocess the text data
sequence = tokenizer.texts_to_sequences([text])
data = pad_sequences(sequence, maxlen=max_length)
 
# Perform the prediction
prediction = model.predict([image_data, numbers_data, data])
