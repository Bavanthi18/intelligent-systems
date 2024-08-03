import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Streamlit app customization with custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Raleway:wght@300;400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
        background-color: #2e2e2e;
        color: white;
    }
    .main-header {
        font-family: 'Raleway', sans-serif;
        color: white;
        text-align: center;
        margin-top: 20px;
        font-size: 2.5em;
    }
    .sidebar-header {
        font-family: 'Raleway', sans-serif;
        color: white;
        font-size: 1.5em;
        text-align: center;
    }
    .prediction-result {
        font-family: 'Roboto', sans-serif;
        color: white;
        font-size: 1.2em;
        text-align: center;
        margin-top: 20px;
    }
    .progress-bar .stProgress {
        background-color: #32CD32;
    }
    .image-container {
        text-align: center;
        margin-bottom: 20px;
    }
    .upload-button {
        font-family: 'Roboto', sans-serif;
        background-color: #4682B4;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    .save-image-checkbox {
        margin-top: 20px;
        color: white;
    }
    .bar-chart, .accuracy-graph {
        margin-top: 20px;
    }
    .confidence-bar {
        color: white;
    }
    .epoch-line {
        color: white;
    }
    .footer {
        text-align: center;
        padding: 20px;
        font-size: 0.9em;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Define header
st.markdown('<h1 class="main-header">Image Classification Model</h1>', unsafe_allow_html=True)

# Load the model
model = load_model(r'C:\Users\Bavanthika\Image_classify.keras')

# Define data categories
data_cat = data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

# Define image dimensions
img_height = 180
img_width = 180

# Sidebar for input widgets
st.sidebar.markdown('<h2 class="sidebar-header">Fruit or Vegetable</h2>', unsafe_allow_html=True)
st.sidebar.write("Upload an image to classify it as a fruit or vegetable.")

# Upload image
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

# Main section
if uploaded_file is not None:
    with st.spinner('Processing...'):
        # Open the uploaded image
        image = Image.open(uploaded_file)

        # Display image
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Option to save the image
        save_option = st.sidebar.checkbox('Save the image?', key='save_image', help='Save the uploaded image to the local directory.')

        if save_option:
            image.save("uploaded_image.png")
            st.sidebar.write("Image saved as 'uploaded_image.png'")

        # Convert the image to a format that TensorFlow can process
        image = image.convert('RGB')
        img_array = np.array(image.resize((img_height, img_width)))
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predict = model.predict(img_array)
        score = tf.nn.softmax(predict[0])
        predicted_label = data_cat[np.argmax(score)]
        accuracy = np.max(score) * 100

        st.markdown(f'<div class="prediction-result">Veg/Fruit in image is <strong>{predicted_label}</strong></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prediction-result">With accuracy of <strong>{accuracy:.2f}%</strong></div>', unsafe_allow_html=True)

        # Display progress
        st.progress(int(accuracy))

        # Display prediction confidence as a bar chart
        st.markdown('<div class="bar-chart">', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.barh(data_cat, predict[0], color='#FF1493')
        ax.set_xlabel('Confidence')
        ax.set_title('Prediction Confidence for Each Category')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.write("Please upload an image file.")

# Accuracy graph (dummy data for illustration)
st.sidebar.markdown('<h2 class="sidebar-header">Model Accuracy Over Epochs</h2>', unsafe_allow_html=True)

# Dummy accuracy data
epochs = range(1, 11)
accuracy = [0.6, 0.65, 0.7, 0.75, 0.78, 0.8, 0.82, 0.85, 0.87, 0.9]

st.markdown('<div class="accuracy-graph">', unsafe_allow_html=True)
fig, ax = plt.subplots()
ax.plot(epochs, accuracy, marker='o', linestyle='-', color='#9400D3')
ax.set_title('Model Accuracy Over Epochs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_xticks(epochs)
ax.set_ylim(0.5, 1.0)
st.sidebar.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Developed with ❤️ by Bavanthika</div>', unsafe_allow_html=True)
