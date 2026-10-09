# BANKING77 Intent Classification

Fine-tune DistilBERT to classify banking customer messages into the 77 BANKING77 intent categories, export the best checkpoint, and run predictions in a Streamlit app.

## Project flow

`BANKING77 -> DistilBERT tokenizer -> fine-tuning -> exported model -> Streamlit prediction`

The training script downloads the official `PolyAI/banking77` dataset through Hugging Face Datasets. It uses the dataset's 10,003 training examples and evaluates against the 3,080 test examples. The exported model includes the tokenizer and intent label mapping.

## Setup

Python 3.10 or newer is recommended. Install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train and export

```bash
python train.py
```

The default run fine-tunes `distilbert-base-uncased` for two epochs and writes the best test-accuracy checkpoint and metrics to `artifacts/banking77-distilbert/`. Training downloads the base model and dataset on first run. To run a quick pipeline smoke test on a smaller training subset:

```bash
python train.py --epochs 1 --max-train-samples 256 --max-test-samples 256 --batch-size 8
```

Use `python train.py --help` to adjust the model, learning rate, batch size, and other training options. A GPU is used automatically when available; CPU training works but is slower.

## Launch the app

After training:

```bash
streamlit run app.py
```

Enter a customer message to see the most likely intent, its confidence, and the next two likely intents. The app reads the local exported model; no hosted inference service is required.

## Example

Input: “I tried to withdraw ₹10,000, but the ATM didn't give me the cash.”

Expected intent family: `cash_withdrawal`.

Model confidence depends on the trained checkpoint and this illustrative output is not guaranteed. BANKING77 is a benchmark dataset; review and validate predictions before using them in customer-facing workflows.