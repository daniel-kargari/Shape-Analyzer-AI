import numpy as np  # Import NumPy for numerical operations
from PIL import Image, ImageDraw  # Import PIL for image creation and manipulation
import random  # Import random for generating random numbers
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import tensorflow as tf  # Import TensorFlow for machine learning
from tensorflow.keras.models import Model  # Import Keras Model class for building models
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Dropout  # Import layers for building the neural network
from tensorflow.keras.utils import to_categorical  # Import utility to convert labels to categorical format
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler  # Import callbacks for training
from concurrent.futures import ThreadPoolExecutor  # Import ThreadPoolExecutor for parallel processing

# Function to create a 100x100 image with random shapes
def create_image():
    image = Image.new('RGB', (100, 100), (255, 255, 255))  # Create a new blank white image
    draw = ImageDraw.Draw(image)  # Create a drawing context

    num_shapes = [random.randint(0, 3) for _ in range(3)]  # Randomly determine the number of rectangles, circles, and triangles
    num_rectangles, num_circles, num_triangles = num_shapes  # Unpack the numbers

    for _ in range(num_rectangles):  # Loop to draw rectangles
        x0, y0, x1, y1 = random.randint(0, 80), random.randint(0, 80), random.randint(5, 20), random.randint(5, 20)
        draw.rectangle([x0, y0, x0 + x1, y0 + y1], outline="black", width=1)

    for _ in range(num_circles):  # Loop to draw circles
        x0, y0, diameter = random.randint(0, 80), random.randint(0, 80), random.randint(5, 20)
        draw.ellipse([x0, y0, x0 + diameter, y0 + diameter], outline="black", width=1)

    for _ in range(num_triangles):  # Loop to draw triangles
        x0, y0, x1, y1, x2, y2 = random.randint(0, 80), random.randint(0, 80), random.randint(5, 20), random.randint(0, 20), random.randint(0, 20), random.randint(5, 20)
        draw.polygon([x0, y0, x0 + x1, y0 + y1, x0 + x2, y0 + y2], outline="black", width=1)

    return image, num_rectangles, num_circles, num_triangles  # Return the image and the counts of each shape

# Function to create and compile a CNN model
def create_model():
    inputs = Input(shape=(100, 100, 1))  # Define the input layer with shape (100, 100, 1) for grayscale images
    x = Conv2D(32, (3, 3), activation='relu')(inputs)  # Add a Conv2D layer with 32 filters and ReLU activation
    x = MaxPooling2D((2, 2))(x)  # Add a MaxPooling2D layer to reduce spatial dimensions
    x = Conv2D(64, (3, 3), activation='relu')(x)  # Add another Conv2D layer with 64 filters
    x = MaxPooling2D((2, 2))(x)  # Add another MaxPooling2D layer
    x = Conv2D(128, (3, 3), activation='relu')(x)  # Add another Conv2D layer with 128 filters
    x = MaxPooling2D((2, 2))(x)  # Add another MaxPooling2D layer
    x = Flatten()(x)  # Flatten the output to create a 1D vector
    x = Dense(256, activation='relu')(x)  # Add a Dense layer with 256 units and ReLU activation
    x = Dropout(0.5)(x)  # Add a Dropout layer to prevent overfitting
    x = Dense(128, activation='relu')(x)  # Add another Dense layer with 128 units

    # Add output layers for each type of shape with 4 units (one for each class) and softmax activation
    output_rectangles = Dense(4, activation='softmax', name='rectangles')(x)
    output_circles = Dense(4, activation='softmax', name='circles')(x)
    output_triangles = Dense(4, activation='softmax', name='triangles')(x)

    model = Model(inputs=inputs, outputs=[output_rectangles, output_circles, output_triangles])  # Create the model

    # Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy metrics
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics={'rectangles': 'accuracy', 'circles': 'accuracy', 'triangles': 'accuracy'})
    return model  # Return the compiled model

# Function to generate the dataset in parallel
def generate_dataset(num_samples):
    images, labels_rectangles, labels_circles, labels_triangles = [], [], [], []  # Initialize lists to store images and labels
    with ThreadPoolExecutor() as executor:  # Use ThreadPoolExecutor for parallel processing
        futures = [executor.submit(create_image) for _ in range(num_samples)]  # Schedule the creation of images
        for future in futures:  # Iterate over the completed futures
            image, num_rectangles, num_circles, num_triangles = future.result()  # Get the result from each future
            images.append(np.array(image.convert('L')))  # Convert the image to grayscale and append to the list
            labels_rectangles.append(num_rectangles)  # Append the rectangle count to the list
            labels_circles.append(num_circles)  # Append the circle count to the list
            labels_triangles.append(num_triangles)  # Append the triangle count to the list

    return images, labels_rectangles, labels_circles, labels_triangles  # Return the lists of images and labels

# Create training data
num_samples = 5000  # Define the number of samples to generate
images, labels_rectangles, labels_circles, labels_triangles = generate_dataset(num_samples)  # Generate the dataset

images = np.array(images).reshape(-1, 100, 100, 1) / 255.0  # Convert images to a NumPy array, reshape, and normalize
labels_rectangles = to_categorical(labels_rectangles, num_classes=4)  # Convert rectangle labels to categorical format
labels_circles = to_categorical(labels_circles, num_classes=4)  # Convert circle labels to categorical format
labels_triangles = to_categorical(labels_triangles, num_classes=4)  # Convert triangle labels to categorical format

# Split the data into training and testing sets (80% training, 20% testing)
train_images, test_images = images[:4000], images[4000:]  # Split images
train_labels_rectangles, test_labels_rectangles = labels_rectangles[:4000], labels_rectangles[4000:]  # Split rectangle labels
train_labels_circles, test_labels_circles = labels_circles[:4000], labels_circles[4000:]  # Split circle labels
train_labels_triangles, test_labels_triangles = labels_triangles[:4000], labels_triangles[4000:]  # Split triangle labels

# Create and train the model
model = create_model()  # Create the model using the create_model function
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Define early stopping to prevent overfitting

# Define a learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr  # Keep the learning rate the same for the first 10 epochs
    else:
        return float(lr * tf.math.exp(-0.1))  # Exponentially decay the learning rate

lr_scheduler = LearningRateScheduler(scheduler)  # Create a learning rate scheduler

# Train the model
model.fit(train_images, [train_labels_rectangles, train_labels_circles, train_labels_triangles],
          epochs=20,  # Train for 20 epochs
          validation_data=(test_images, [test_labels_rectangles, test_labels_circles, test_labels_triangles]),  # Use testing data for validation
          callbacks=[early_stopping, lr_scheduler])  # Use early stopping and learning rate scheduler callbacks

# Function to visualize the image and predictions
def visualize_predictions(image, predictions):
    plt.imshow(image, cmap='gray')  # Display the image in grayscale
    plt.title(f"Rectangles: {predictions[0]}, Circles: {predictions[1]}, Triangles: {predictions[2]}")  # Add a title with predictions
    plt.show()  # Show the plot

# Generate a new image and make predictions
test_image, _, _, _ = create_image()  # Generate a new test image
test_image_array = np.array(test_image.convert('L')).reshape(-1, 100, 100, 1) / 255.0  # Convert the test image to the required format
predictions = model.predict(test_image_array)  # Make predictions using the trained model

# Extract the predicted counts of each shape
predicted_counts_rectangles = np.argmax(predictions[0], axis=1)[0]  # Find the index of the maximum value for rectangles
predicted_counts_circles = np.argmax(predictions[1], axis=1)[0]  # Find the index of the maximum value for circles
predicted_counts_triangles = np.argmax(predictions[2], axis=1)[0]  # Find the index of the maximum value for triangles

# Visualize the image and predictions
visualize_predictions(test_image, [predicted_counts_rectangles, predicted_counts_circles, predicted_counts_triangles])  # Display the results
