# CLAY Model: EuroSAT LoRA Fine-Tuning & Edge Deployment

This project fine-tunes the CLAY foundation model with LoRA for EuroSAT land-use classification, then deploys lightweight inference on Raspberry Pi CPU.

---

## 1. Project Structure

### Model and Training
- `train_lora.py`: Train the LoRA adapter on EuroSAT.
- `prepare_data.py`: Download and prepare EuroSAT RGB data.
- `configs/metadata.yaml`: Required CLAY model metadata.
- `requirements_lora.txt`: Dependencies for training and validation.

### Evaluation and Testing
- `test_inference.py`: Batch validation and confusion matrix generation.
- `extract_sample_images.py`: Pick sample images for quick manual checks.
- `outputs/`: Stores confusion matrix and evaluation artifacts.
- `sample_tests/`: Local images for single-image inference tests.

### Raspberry Pi (Edge Inference)
- `pi_inference.py`: CPU inference script for Raspberry Pi.
- `requirements_pi.txt`: Minimal inference-only dependencies.

---

## 2. PC / GPU Usage (Training & Validation)

### How to Train
To start LoRA fine-tuning:

```bash
python CLAY/train_lora.py
```

### How to Evaluate (Batch)
To run inference on the entire validation set and generate a confusion matrix:

```bash
python CLAY/test_inference.py
```

---

## 3. Raspberry Pi Deployment (Single Image Inference)

Follow these steps to run the model on a Raspberry Pi CPU.

### Step 1: Environment Setup
Install lightweight libraries required for CPU inference:

```bash
pip install -r requirements_pi.txt
```

### Step 2: Select an Image to Test
Because `pi_inference.py` is designed for one image at a time, choose a test image in `sample_tests/` and run:

```bash
python CLAY/pi_inference.py --image_path CLAY/sample_tests/<your_image>.jpg
```

### Step 3: Execute
Run single-image inference in terminal:

```bash
python CLAY/pi_inference.py
```

> Note: First-time model loading can take around 30 seconds on Pi-class CPU devices.

---

## 4. Performance and Insights

- Validation accuracy: ~96.8%.
- Edge target: ARM-based CPU deployment (Raspberry Pi).
- Common confusion pairs: similar-looking classes such as annual crop vs permanent crop.