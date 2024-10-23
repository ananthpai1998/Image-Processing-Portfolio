import plotly.graph_objects as go
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from functools import partial
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from pycpd import DeformableRegistration
from IPython.display import clear_output, display
from scipy.ndimage import gaussian_filter
import os
from PIL import Image


def apply_gaussian_smoothing(data, kernel_size=(5, 5), sigma=3):
    data_reshaped = data.astype(np.float32)
    smoothed_data = cv2.GaussianBlur(data_reshaped, kernel_size, sigma)
    return smoothed_data


def generate_data(apply_blur=True):
    x = np.linspace(-5, 5, 23)
    y = np.linspace(-5, 5, 23)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)
    source = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    noise_dx = np.random.normal(loc=0.0, scale=5, size=x.shape)
    noise_dy = np.random.normal(loc=0.0, scale=5, size=y.shape)
    noise_dz = np.random.normal(loc=0.0, scale=5, size=z.shape)

    x_noisy = x + noise_dx
    y_noisy = y + noise_dy
    z_noisy = z + noise_dz

    if apply_blur:
        smooth_dx = apply_gaussian_smoothing(x_noisy, kernel_size=(5, 5), sigma=10)
        smooth_dy = apply_gaussian_smoothing(y_noisy, kernel_size=(5, 5), sigma=10)
        smooth_dz = apply_gaussian_smoothing(z_noisy, kernel_size=(5, 5), sigma=10)

        target = np.column_stack((smooth_dx.flatten(), smooth_dy.flatten(), smooth_dz.flatten()))
    else:
        target = np.column_stack((x_noisy.flatten(), y_noisy.flatten(), z_noisy.flatten()))

    return source, target


def plot_3d_surface_interactive_surface(points, title="3D Surface Plot", rotation_speed=0.1, rotation_duration=10):
    x = points[:, 0].reshape(23, 23)
    y = points[:, 1].reshape(23, 23)
    z = points[:, 2].reshape(23, 23)
    
    fig = go.Figure(data=[go.Surface(
        z=z,
        x=x,
        y=y,
        colorscale='Viridis'
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            zaxis=dict(range=[-5, 5]) 
        ),
        updatemenus=[dict(type="buttons", showactive=False, buttons=[dict(label="Play",
            method="animate",
            args=[None, dict(frame=dict(duration=15, redraw=True), fromcurrent=True, mode='immediate')])])])

    frames = []
    for t in np.arange(0, rotation_duration, rotation_speed):
        camera = dict(eye=dict(x=2*np.cos(t), y=2*np.sin(t), z=0.5)
        )
        frames.append(go.Frame(layout=dict(scene_camera=camera)))
    fig.frames = frames
    fig.show()


def pycpd_registration(source, target):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    visualize_callback = partial(visualize_plot, ax=ax)
    reg = DeformableRegistration(X=target, Y=source)
    reg.register(callback=visualize_callback)  
    transformed_source = reg.transform_point_cloud(source)
    return transformed_source


def visualize_plot(iteration, error, X, Y, ax):
    clear_output(wait=True)  
    
    ax.cla()
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='red', label='Target', s=20)  
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='blue', label='Source', s=20)
    mse = np.mean(np.square(X - Y))
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(f'3D Surface Registration')
    ax.text2D(0.05, 0.95, f'Iteration: {iteration}, Error: {mse:.4f}', 
              transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    # Explicitly reference the figure from the axes
    fig = ax.get_figure()
    
    # Draw the plot before saving
    fig.canvas.draw_idle()  # Ensures canvas updates before saving
    plt.pause(0.01)

    if not os.path.exists(r'gif_folder'):
        os.makedirs(r'gif_folder')
    
    # Save the figure as an image
    fig.savefig(f"gif_folder/frame_{iteration:03d}.png")
    
    # Display the figure in the notebook
    display(fig)

def calculate_error(transformed_source, target):
    mse = np.mean(np.square(transformed_source - target))
    return mse

def visualize_interactive_plot(source, target, transformed_source):
    trace1 = go.Scatter3d(
        x=target[:, 0], y=target[:, 1], z=target[:, 2],
        mode='markers',
        marker=dict(size=3, color='red'),
        name='Target (Scatter)',
        showlegend=True 
    )

    trace2 = go.Scatter3d(
        x=source[:, 0], y=source[:, 1], z=source[:, 2],
        mode='markers',
        marker=dict(size=3, color='blue'),
        name='Source (Scatter)',
        showlegend=True
    )

    trace3 = go.Scatter3d(
        x=transformed_source[:, 0], y=transformed_source[:, 1], z=transformed_source[:, 2],
        mode='markers',
        marker=dict(size=3, color='green'),
        name='Transformed Source (Scatter)',
        showlegend=True
    )

    surface_target = go.Surface(
        x=target[:, 0].reshape(23, 23),  
        y=target[:, 1].reshape(23, 23),
        z=target[:, 2].reshape(23, 23),
        colorscale='Reds',
        opacity=0.6,
        name='Target (Surface)',
        showscale=False  
    )

    surface_source = go.Surface(
        x=source[:, 0].reshape(23, 23),  
        y=source[:, 1].reshape(23, 23),
        z=source[:, 2].reshape(23, 23),
        colorscale='Blues',
        opacity=0.6,
        name='Source (Surface)',
        showscale=False 
    )

    surface_transformed = go.Surface(
        x=transformed_source[:, 0].reshape(23, 23),
        y=transformed_source[:, 1].reshape(23, 23),
        z=transformed_source[:, 2].reshape(23, 23),
        colorscale='Greens',
        opacity=0.6,
        name='Transformed Source (Surface)',
        showscale=False  
    )

    data = [trace1, trace2, trace3, surface_target, surface_source, surface_transformed]
    layout = go.Layout(
        title="3D Surface Registration: Target, Source, and Transformed Source (Scatter and Surface)",
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis"
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()

class DeformableRegistration_Custom:
    def __init__(self, target, source):
        self.target = tf.constant(target, dtype=tf.float32)
        self.source = tf.Variable(source, dtype=tf.float32)
        
        self.scaling_matrix = self.get_scaling_matrix(3)
        self.rotation_matrix = self.get_rotation_matrix(np.pi / 2)
        self.translation_matrix = self.get_translation_matrix(3)

        self.combined_transform = self.get_combined_transform()

        self.model = models.Sequential([
            layers.Dense(100, activation='relu', input_shape=(3,)),
            layers.Dense(3)
        ])
        
        self.optimizer = optimizers.Adam(learning_rate=0.01)
        self.loss_fn = losses.MeanSquaredError()

    @staticmethod
    def get_scaling_matrix(scale):
        return tf.constant([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=tf.float32)

    @staticmethod
    def get_rotation_matrix(angle):
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        return tf.constant([[cos_val, -sin_val, 0], [sin_val, cos_val, 0], [0, 0, 1]], dtype=tf.float32)

    @staticmethod
    def get_translation_matrix(shift):
        return tf.constant([[1, 0, shift], [0, 1, shift], [0, 0, 1]], dtype=tf.float32)

    def get_combined_transform(self):
        return tf.matmul(tf.matmul(self.scaling_matrix, self.rotation_matrix), self.translation_matrix)
    
    @staticmethod
    def gaussian_weight(size, sigma=1.0):
        gaussian = np.ones((size, size))
        gaussian = gaussian_filter(gaussian, sigma=sigma)
        return tf.constant(gaussian / np.sum(gaussian), dtype=tf.float32)

    def apply_transformation(self, source):
        return tf.matmul(source, self.combined_transform)

    def register(self, callback, iterations=1000):
        for i in range(iterations):
            with tf.GradientTape() as tape:
                transformed_input = self.apply_transformation(self.source)
                transformed_source = self.model(transformed_input, training=True)
                loss = self.loss_fn(self.target, transformed_source)
            grads = tape.gradient(loss, self.model.trainable_variables + [self.source])
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables + [self.source]))
            callback(i, loss.numpy(), self.target.numpy(), transformed_source.numpy())

        return self.model.layers[-1].kernel.numpy()
    

class DeformableRegistration_Custom2:
    def __init__(self, target, source):
        self.target = tf.constant(target, dtype=tf.float32)
        self.source = tf.Variable(source, dtype=tf.float32)

        # Make scale, rotation angle, and translation shift trainable variables
        self.scale = tf.Variable(3.0, dtype=tf.float32)   # Learnable scaling factor
        self.rotation_angle = tf.Variable(np.pi / 2, dtype=tf.float32)  # Learnable rotation angle
        self.translation_shift = tf.Variable(3.0, dtype=tf.float32)  # Learnable translation shift

        # Initialize the model for additional deformation learning
        self.model = models.Sequential([
            layers.Dense(100, activation='relu', input_shape=(3,)),
            layers.Dense(3)
        ])
        
        self.optimizer = optimizers.Adam(learning_rate=0.01)
        self.loss_fn = losses.MeanSquaredError()

    def get_dynamic_scaling_matrix(self):
        return tf.convert_to_tensor([[self.scale, 0, 0], 
                                     [0, self.scale, 0], 
                                     [0, 0, 1]], dtype=tf.float32)

    def get_dynamic_rotation_matrix(self):
        cos_val = tf.cos(self.rotation_angle)
        sin_val = tf.sin(self.rotation_angle)
        return tf.convert_to_tensor([[cos_val, -sin_val, 0], 
                                     [sin_val, cos_val, 0], 
                                     [0, 0, 1]], dtype=tf.float32)

    def get_dynamic_translation_matrix(self):
        return tf.convert_to_tensor([[1, 0, self.translation_shift], 
                                     [0, 1, self.translation_shift], 
                                     [0, 0, 1]], dtype=tf.float32)

    def get_combined_transform(self):
        # Dynamically compute the combined transformation based on the current values
        scaling_matrix = self.get_dynamic_scaling_matrix()
        rotation_matrix = self.get_dynamic_rotation_matrix()
        translation_matrix = self.get_dynamic_translation_matrix()

        return tf.matmul(tf.matmul(scaling_matrix, rotation_matrix), translation_matrix)

    def apply_transformation(self, source):
        combined_transform = self.get_combined_transform()
        return tf.matmul(source, combined_transform)

    def register(self, callback, iterations=1000):
        for i in range(iterations):
            with tf.GradientTape() as tape:
                transformed_input = self.apply_transformation(self.source)
                transformed_source = self.model(transformed_input, training=True)
                loss = self.loss_fn(self.target, transformed_source)

            # Calculate gradients for both the neural network and the matrices (scale, rotation, translation)
            grads = tape.gradient(loss, self.model.trainable_variables + 
                                  [self.scale, self.rotation_angle, self.translation_shift, self.source])
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables + 
                                               [self.scale, self.rotation_angle, self.translation_shift, self.source]))
            callback(i, loss.numpy(), self.target.numpy(), transformed_source.numpy())

        return self.model.layers[-1].kernel.numpy()
    

def create_gif_with_variable_duration(folder_path, output_gif):
    # Get all the image files from the folder
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    output_dir = os.path.dirname(output_gif)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List to store all image frames
    frames = []
    durations = []

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        frame = Image.open(image_path)
        frames.append(frame)

        # Set duration: 0.1 seconds for first 120 frames, 0.05 for the rest
        if i < 120:
            durations.append(100)  # 0.1 second
        else:
            durations.append(25)   # 0.05 second

    # Save frames as a GIF with variable durations
    if frames:
        frames[0].save(output_gif, format='GIF', append_images=frames[1:], save_all=True, 
                       duration=durations, loop=0)

