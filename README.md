# ct-reconstruction-dl
# CT Reconstruction via Deep Learning

**Author**: Lakshan Mdushanka  
**Description**: A neural network pipeline for reconstructing CT images from sinogram projections using a custom L2+gradient loss and hybrid architecture. Trained on paired sinogram-image data for improved image quality.

## Setup
1. Clone: `git clone https://github.com//DLakshanMadushnka/ct-reconstruction-dl.git`
2. Install: `pip install -r requirements.txt`
3. Place data (.npy files) in `data/` and update `config.py`.
4. Run: `python main.py`

## Key Features
- Hybrid model: Dense embedding + convolutional refinement with residuals.
- Metrics: MSE, PSNR, SSIM evaluation.
- GPU-optimized with mixed precision.

## Results
- Typical PSNR: 25-30 dB on validation.
- See `outputs/` for models, reconstructions, and history plots.

## Usage
- Load sinograms as flattened vectors.
- Predict: `model.predict(sinogram_input)`.

## License
MIT
