# Personal PDF Chatbot (Local RAG)

A small local Streamlit app for asking questions about one PDF at a time. It extracts text page-by-page, creates overlapping chunks, retrieves relevant chunks with TF-IDF, and asks a local Ollama model to answer using those excerpts only.

## Requirements

- Windows 10/11
- Python 3.11 recommended
- Ollama for Windows
- About 2–4 GB of free disk space recommended for Python packages and the model; RAM use varies.

## 1. Install Python dependencies

Open Command Prompt in this folder:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, install Python 3.11 and select "Add Python to PATH", or use `python -m venv .venv`.

## 2. Install and start Ollama

Install Ollama for Windows from https://ollama.com/download/windows

Then open Command Prompt and run:

```bat
ollama pull qwen2.5:0.5b
ollama run qwen2.5:0.5b
```

After the model starts, you can type `/bye` to exit the interactive session. Ollama normally stays available in the background. If the app cannot connect, launch Ollama from the Start menu and retry.

**Low-memory option:** `qwen2.5:0.5b` is the recommended starting point for a 4 GB RAM laptop. Even this may be slow depending on Windows background usage. Close other applications before running. Do not start with a 7B model.

## 3. Launch the app

Double-click `run.bat`, or run:

```bat
.venv\Scripts\activate
python -m streamlit run app.py
```

Streamlit opens a local address, usually http://localhost:8501.

## How it works

1. PyMuPDF extracts text from each PDF page.
2. Each page is split into overlapping chunks. Chunks retain their original page number.
3. TF-IDF vectorization and cosine similarity retrieve likely relevant chunks without a separate embedding model.
4. The retrieved excerpts are sent to the local Ollama model.
5. The prompt instructs the model to use only those excerpts and cite page numbers.

## Grounding and limitations

- Retrieval uses lexical TF-IDF matching, not a neural embedding model. It is lightweight but may miss paraphrases and some semantically related questions.
- A minimum similarity threshold is used to abstain when no relevant passage is found. Adjust it in the sidebar: higher means more conservative.
- LLM instructions and retrieval thresholds reduce unsupported answers but cannot mathematically guarantee zero hallucinations. Always inspect the displayed evidence.
- Scanned/image-only PDFs generally need OCR before upload. This app intentionally does not run a large OCR engine to keep the installation light.
- The app processes one uploaded PDF at a time. Uploading another PDF replaces the index and clears the conversation.
- PDF text is sent to the local Ollama endpoint at `127.0.0.1`; no cloud model API is used by this app. Installing software and downloading the model requires internet access.
- Page numbers are PDF page positions (first page is Page 1), not necessarily printed page labels in the document.

## Troubleshooting

**`ollama` is not recognized:** Reopen Command Prompt after installing Ollama. Check installation in a new terminal with `ollama --version`.

**Model not found:** Run `ollama pull qwen2.5:0.5b`.

**Ollama connection failed:** Open the Ollama desktop app. Confirm `http://127.0.0.1:11434/api/tags` responds in a browser.

**Out of memory / very slow:** Close browsers and other applications, restart Ollama, and use the 0.5B model. Consider retrieval-only mode as a future fallback if the machine cannot run inference.

**No text found:** The PDF may be scanned or image-only. OCR it with a trusted local OCR tool and save a searchable PDF.

## Suggested test questions

1. Ask a question whose answer appears clearly in the PDF.
2. Ask for a specific number, date, or name and inspect the cited page.
3. Ask a question whose answer is not in the PDF. The expected response is: "The answer could not be found in the uploaded document."
4. Ask an unrelated question.
5. Try a question phrased differently from the PDF wording to test retrieval limitations.
