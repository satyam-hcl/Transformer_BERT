# BANKING77 Intent Classification

Fine-tune DistilBERT to classify banking customer messages into the 77 BANKING77 intent categories, export the best checkpoint, and run predictions in a Streamlit app.

## Project flow

`BANKING77 -> fine-tuned transformer -> Streamlit prediction`

The training script downloads the official `PolyAI/banking77` dataset through Hugging Face Datasets. It uses the dataset's 10,003 training examples and evaluates against the 3,080 test examples. The exported model includes the tokenizer and intent label mapping. For deployments where the local export is not present, the app loads a public BANKING77 fine-tuned model from Hugging Face Hub.

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

Run the app locally:

```bash
streamlit run app.py
```

The app uses `artifacts/banking77-distilbert/` when a local trained export exists. Otherwise, it downloads `mrm8488/distilroberta-finetuned-banking77` from Hugging Face Hub, so Streamlit Community Cloud does not need a model artifact committed to the repository. The first launch needs internet access and may take a little longer while model files are downloaded.

To use your own model uploaded to Hugging Face Hub, set the `BANKING77_MODEL_ID` environment variable (or add it under **App settings → Secrets** in Streamlit Community Cloud) to its repository ID, for example `your-account/your-banking77-model`. The repository must contain a Transformers sequence-classification model and tokenizer.

Enter a customer message to see the most likely intent, its confidence, and the next two likely intents. No hosted inference API is used; the model runs in the app process.

## Example

Input: “I tried to withdraw ₹10,000, but the ATM didn't give me the cash.”

Expected intent family: `cash_withdrawal`.

Model confidence depends on the trained checkpoint and this illustrative output is not guaranteed. BANKING77 is a benchmark dataset; review and validate predictions before using them in customer-facing workflows.