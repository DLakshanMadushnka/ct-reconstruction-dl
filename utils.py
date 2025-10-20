import tensorflow as tf
from tensorflow.keras import backend as K
from skimage.metrics import structural_similarity as ssim

def l2_and_gradient_loss(y_true, y_pred):
    """Custom loss: L2 + total variation (gradient) penalty."""
    l2_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Gradient penalty (total variation)
    grad_x = K.abs(y_pred[:, :, 1:] - y_pred[:, :, :-1])
    grad_y = K.abs(y_pred[:, 1:, :] - y_pred[:, :-1, :])
    tv_loss = K.mean(grad_x) + K.mean(grad_y)
    
    return l2_loss + 0.1 * tv_loss  # Weight TV term

def psnr(y_true, y_pred):
    """Peak Signal-to-Noise Ratio."""
    return tf.reduce_mean(20 * tf.math.log(255.0 / tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3]))))

def compute_ssim(y_true, y_pred):
    """Compute SSIM (batch-averaged)."""
    ssim_vals = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return tf.reduce_mean(ssim_vals)
