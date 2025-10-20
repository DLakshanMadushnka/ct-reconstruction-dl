import os

class Config:
    # Paths
    DATA_PATH = '/content/gdrive/MyDrive/'  # Adjust for local/Colab
    X_TRAIN_PATH = os.path.join(DATA_PATH, 'X_train.npy')
    Y_TRAIN_PATH = os.path.join(DATA_PATH, 'y_train.npy')
    X_VAL_PATH = os.path.join(DATA_PATH, 'X_test.npy')
    Y_VAL_PATH = os.path.join(DATA_PATH, 'y_test.npy')
    OUTPUT_PATH = os.path.join(DATA_PATH, 'ct_reconstruction_outputs/')
    MODEL_PATH = os.path.join(OUTPUT_PATH, 'best_ct_model.keras')
    
    # Data/Model
    USER_SIZE = 128  # Assumed sinogram patch size (user*user)
    BATCH_SIZE = 32
    EPOCHS = 300
    LEARNING_RATE = 1e-4
    PATIENCE = 20  # For early stopping
    
    # Reproducibility
    SEED = 42
