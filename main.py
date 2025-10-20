import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils import l2_and_gradient_loss, psnr, compute_ssim  # Custom utils

from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seeds
np.random.seed(Config.SEED)
tf.random.set_seed(Config.SEED)

# Enable mixed precision for GPU efficiency
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def load_data():
    """Load and preprocess data."""
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    
    x_train = np.load(Config.X_TRAIN_PATH).astype(np.float32)
    y_train = np.load(Config.Y_TRAIN_PATH).astype(np.float32)
    x_val = np.load(Config.X_VAL_PATH).astype(np.float32)
    y_val = np.load(Config.Y_VAL_PATH).astype(np.float32)
    
    # Reshape sinograms: Assume flattened 2*user*user -> (batch, 2*user*user)
    # Images: (batch, user, user, 1)
    x_train = x_train.reshape(-1, 2 * Config.USER_SIZE * Config.USER_SIZE)
    x_val = x_val.reshape(-1, 2 * Config.USER_SIZE * Config.USER_SIZE)
    y_train = y_train.reshape(-1, Config.USER_SIZE, Config.USER_SIZE, 1)
    y_val = y_val.reshape(-1, Config.USER_SIZE, Config.USER_SIZE, 1)
    
    # Normalize: Sinograms to [-1,1], images to [0,1]
    x_train = (x_train / np.max(np.abs(x_train))) * 2 - 1
    x_val = (x_val / np.max(np.abs(x_val))) * 2 - 1
    y_train = y_train / np.max(np.abs(y_train))
    y_val = y_val / np.max(np.abs(y_val))
    
    logger.info(f"Loaded: Train {x_train.shape}, Val {x_val.shape}")
    return x_train, y_train, x_val, y_val

def build_model():
    """Build hybrid dense-conv CT reconstruction model."""
    inputs = layers.Input(shape=(2 * Config.USER_SIZE * Config.USER_SIZE,))
    
    # Dense layers for sinogram embedding
    x = layers.Dense(Config.USER_SIZE * Config.USER_SIZE, activation='tanh')(inputs)
    x = layers.Dense(Config.USER_SIZE * Config.USER_SIZE, activation='tanh')(x)
    x = layers.Dense(Config.USER_SIZE * Config.USER_SIZE, activation='tanh')(x)
    x = layers.Reshape((Config.USER_SIZE, Config.USER_SIZE, 1))(x)
    
    # Convolutional refinement (with residuals for stability)
    for filters, kernel in [(64, 3), (64, 3), (64, 5), (64, 5), (64, 5)]:
        res = x
        x = layers.Conv2D(filters, kernel, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        if x.shape[-1] == res.shape[-1]:
            x = layers.Add()([x, res])  # Residual connection
    
    # Output layer
    outputs = layers.Conv2DTranspose(1, (7, 7), padding='same', activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss=l2_and_gradient_loss,
        metrics=[psnr]  # Custom PSNR metric
    )
    return model

def train_and_evaluate():
    """Train model and evaluate."""
    x_train, y_train, x_val, y_val = load_data()
    
    model = build_model()
    logger.info(model.summary())
    
    # Datasets for efficiency
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    callbacks = [
        ModelCheckpoint(Config.MODEL_PATH, save_best_only=True, monitor='val_loss'),
        EarlyStopping(patience=Config.PATIENCE, monitor='val_loss', restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=10, monitor='val_loss')
    ]
    
    history = model.fit(
        train_ds, epochs=Config.EPOCHS, validation_data=val_ds,
        callbacks=callbacks, verbose=1
    )
    
    # Evaluation
    y_pred = model.predict(val_ds, verbose=0)
    mse = np.mean((y_val - y_pred) ** 2)
    psnr_val = np.mean([compare_psnr(y_val[i].squeeze(), y_pred[i].squeeze()) for i in range(len(y_val))])
    ssim_val = np.mean([compute_ssim(y_val[i], y_pred[i]) for i in range(len(y_val))])  # Scalar SSIM
    
    logger.info(f"Final Metrics - MSE: {mse:.4f}, PSNR: {psnr_val:.4f} dB, SSIM: {ssim_val:.4f}")
    
    # Visualizations
    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(y_val[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        plt.title(f'GT {i+1}')
        plt.axis('off')
        plt.subplot(1, 5, i+6)
        plt.imshow(y_pred[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        plt.title(f'Recon {i+1}\nPSNR: {compare_psnr(y_val[i].squeeze(), y_pred[i].squeeze()):.2f}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_PATH, 'reconstructions.png'), dpi=150)
    plt.close()
    
    # Loss plot
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(Config.OUTPUT_PATH, 'training_history.png'), dpi=150)
    plt.close()
    
    logger.info(f"Training complete. Outputs saved to {Config.OUTPUT_PATH}")

if __name__ == "__main__":
    # Mount Drive in Colab: from google.colab import drive; drive.mount('/content/gdrive')
    train_and_evaluate()
