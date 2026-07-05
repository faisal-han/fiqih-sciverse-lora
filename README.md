# fiqih-sciverse-lora

<p align="center">
  <img src="https://img.shields.io/badge/Model-Llama3--8B--Instruct-blue?style=for-the-badge&logo=meta" alt="Llama3">
  <img src="https://img.shields.io/badge/Framework-Unsloth--PEFT-orange?style=for-the-badge" alt="Unsloth">
  <img src="https://img.shields.io/badge/Method-LoRA--4bit-green?style=for-the-badge" alt="LoRA">
</p>

## Deskripsi Proyek
**fiqih-sciverse-lora** adalah proyek eksperimen *Fine-Tuning* Large Language Model (LLM) berbasis Meta Llama 3 (8B) menggunakan metode **LoRA (Low-Rank Adaptation)** dan kuantisasi **4-bit (QLoRA)** via Unsloth. 

Tujuan utama dari proyek ini adalah membangun model AI yang mampu menjawab persoalan hukum Islam (Fiqih) secara runut dan objektif, sekaligus mengintegrasikannya dengan **analisis sains empiris (fisika, biofisika, hidrologi, dan mikrobiologi)** yang valid untuk menghindari pendekatan pseudosains (*cocokologi*).

* **Model Base:** `unsloth/llama-3-8b-Instruct-bnb-4bit`
* **Dataset Resmi:** Available on Hugging Face Datasets
* **Weights/Adapter:** Released on Hugging Face Models: [handvi/fiqih-sciverse-lora](https://huggingface.co/handvi/fiqih-sciverse-lora)

---

## Tech Stack & Metode
* **Fine-tuning Engine:** [Unsloth](https://github.com/unslothai/unsloth) (Mempercepat training hingga 2x lebih hemat VRAM pada GPU T4).
* **Framework:** PyTorch, Hugging Face Transformers, TRL (SFTTrainer), dan PEFT.
* **Dataset Format:** Structured JSON containing `instruction` and `output` (Synthesis of classical Fiqih Turats & peer-reviewed science papers).

---

## 📂 Struktur Repositori
* `dataset_fiqih.json`: Dataset latih awal yang memisahkan wilayah dogma hukum (Fiqih) dan wilayah empiris (Sains).
* `fiqih-sciverse.ipynb`: Jupyter Notebook / Google Colab script yang digunakan untuk memproses pipeline training dari loading model hingga push ke Hugging Face Hub.

---

## Cara Menggunakan Model (Inference)

Untuk memanggil model *adapter* ini di Google Colab atau server lokal Anda, pastikan telah menginstal `unsloth` terlebih dahulu, lalu jalankan script Python berikut:

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "handvi/fiqih-sciverse-lora", # Memanggil adapter dari Hugging Face
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Template Prompt Latihan
prompt_template = """Di bawah ini adalah pertanyaan seputar hukum Islam (Fiqih) beserta analisis sains pendukungnya. Berikan jawaban yang runut, ilmiah, dan objektif.

### Pertanyaan:
{}

### Jawaban:
"""

pertanyaan = "Bagaimana hukum bersuci menggunakan air laut menurut fiqih, dan bagaimana analisis sains yang mendukungnya?"
inputs = tokenizer([prompt_template.format(pertanyaan)], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 500, use_cache = True)
print(tokenizer.batch_decode(outputs)[0])
