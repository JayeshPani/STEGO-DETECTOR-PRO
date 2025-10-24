🧬 STEGO DETECTOR PRO

Deep Learning–Based Steganography Detection using EfficientNet-B0

⸻

🚀 Overview

Stego Detector Pro is an advanced AI-powered tool built with EfficientNet-B0 to detect the presence of hidden data (steganography) inside digital images.
It provides a modern Gradio web interface with adjustable detection parameters such as threshold, logit bias, and temperature, allowing real-time, interpretable forensics.

The model was trained on the ALASKA2 dataset, a benchmark dataset for JPEG steganalysis, and achieves reliable distinction between clean and stego images through fine-tuned calibration.

⸻

🖼️ Key Features
	•	⚡ EfficientNet-B0 Backbone — Lightweight and high-accuracy image feature extraction.
	•	🧠 Real-time Gradio Interface — Upload images and visualize predictions instantly.
	•	🎛️ Dynamic Controls — Adjust Decision Threshold, Logit Bias, and Temperature for precise calibration.
	•	💡 Visual Confidence Bar — Color-coded prediction confidence for intuitive analysis.
	•	🔬 Device Auto-Selection — Automatically runs on CUDA / MPS / CPU depending on availability.
	•	🧰 Extensible Pipeline — Modular code structure for retraining and fine-tuning on new datasets.

⸻

🗂️ Project Structure

stego-efficientnet/
│
├── src/
│   ├── app_gradio.py       # Main Gradio web app
│   ├── infer.py            # Command-line inference
│   ├── model.py            # Model definition (EfficientNet-B0)
│   ├── datasets.py         # Dataset & dataloader utilities
│   ├── transforms.py       # Image preprocessing and augmentations
│   ├── train.py            # Training pipeline
│   └── utils.py            # Helper functions
│
├── configs/
│   ├── base.yaml
│   ├── train_full.yaml
│   ├── overfit.yaml
│   └── overfit_true.yaml
│
├── outputs/
│   └── checkpoints/        # Model weights (add efnb0_best.pth here)
│
├── scripts/                # Utility scripts (data splits, etc.)
├── requirements.txt        # Dependencies list
├── README.md               # Project documentation
├── LICENSE                 # License information
└── stego-efficientnet.zip  # Complete project archive


⸻

⚙️ Installation & Setup

1️⃣ Download

Simply download the stego-efficientnet.zip and extract it.

2️⃣ Create a Virtual Environment

python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Run the Gradio Web App

python src/app_gradio.py

Then open the provided URL in your browser (e.g. http://127.0.0.1:7860).

⸻

🧩 Detection Parameters Explained

Parameter Description
Decision Threshold	Defines the cutoff for classifying an image as stego or clean.
Logit Bias	Adjusts the model’s output bias to compensate for dataset skew.
Temperature	Controls the softness of the probability distribution (lower = sharper).


⸻

🧠 Model Notes
	•	Trained on ALASKA2 dataset (Clean vs Stego JPEG pairs).
	•	Uses Binary Cross-Entropy Loss and CosineLR scheduler.
	•	Fine-tuned EfficientNet-B0 with data augmentation (Normalize, Resize, RandomCrop).
	•	Compatible with PyTorch ≥ 2.0 and Gradio ≥ 4.0.

⸻

⚡ Sample Output

Image	Prediction	Confidence
		CLEAN		  52.8 %
	  STEGANOGRAPHY	  98.7 %


⸻

⚠️ Important Notes
	•	Large model files (like efnb0_best.pth) and datasets are excluded from GitHub due to size restrictions.
	•	To run detection, place your trained checkpoint under:

outputs/checkpoints/efnb0_best.pth


	•	The project supports macOS (MPS), Windows, and Linux (CUDA) seamlessly.

⸻

📜 License

This project is licensed under the MIT License — see the LICENSE file for details.

⸻

👤 Author

Jayesh Pani
🧑‍💻 VIT Vellore — Integrated M.Tech CSE Core (2028)
🔗 LinkedIn
📧 jayeshpani14@gmail.com

⸻

💬 Acknowledgements
	•	Dataset: ALASKA2 Steganalysis Dataset (Kaggle)
	•	Model: EfficientNet-B0 (Google Research)
	•	UI: Built using Gradio
	•	Framework: PyTorch

