import streamlit as st
from pdf_processor import extract_pdf_pages, make_chunks
from retriever import LocalRetriever
from llm import ollama_chat, check_ollama, DEFAULT_MODEL

st.set_page_config(page_title="Personal PDF Chatbot", page_icon="📄", layout="wide")

st.title("📄 Personal PDF Chatbot")
st.caption("Ask questions about one PDF. Answers are grounded in retrieved passages from your document.")

ABSTAIN = "The answer could not be found in the uploaded document."

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Ollama model", value=DEFAULT_MODEL, help="For a 4 GB RAM laptop, start with qwen2.5:0.5b.")
    top_k = st.slider("Passages to retrieve", min_value=2, max_value=5, value=3)
    min_score = st.slider(
        "Minimum retrieval score",
        min_value=0.01,
        max_value=0.30,
        value=0.06,
        step=0.01,
        help="Higher values are more conservative but may abstain more often."
    )
    st.markdown("---")
    st.caption("Privacy: the PDF is processed in this app. Text is sent only to your local Ollama server, not to a cloud AI API.")

if "pdf_fingerprint" not in st.session_state:
    st.session_state.pdf_fingerprint = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

uploaded = st.file_uploader("Upload one PDF document", type=["pdf"], accept_multiple_files=False)

if uploaded is None:
    if st.session_state.pdf_name:
        st.info("Your previous document is no longer selected. Upload it again to continue.")
    st.markdown("### How to use")
    st.markdown("1. Upload a text-based PDF.\n2. Wait for indexing to finish.\n3. Ask a question in the chat box.\n4. Expand **Evidence used** to inspect source passages and page numbers.")
    st.stop()

pdf_bytes = uploaded.getvalue()
import hashlib
fingerprint = hashlib.sha256(pdf_bytes).hexdigest()

if fingerprint != st.session_state.pdf_fingerprint:
    with st.spinner("Reading and indexing the PDF locally..."):
        try:
            pages = extract_pdf_pages(pdf_bytes)
            chunks = make_chunks(pages)
            if not chunks:
                st.error("No extractable text was found. This may be a scanned PDF. Run OCR on it first, then upload the searchable PDF.")
                st.stop()
            st.session_state.chunks = chunks
            st.session_state.retriever = LocalRetriever(chunks)
            st.session_state.pdf_fingerprint = fingerprint
            st.session_state.pdf_name = uploaded.name
            st.session_state.messages = []
            st.session_state.page_count = len(pages)
        except Exception as exc:
            st.error(f"Could not read this PDF: {exc}")
            st.stop()

st.success(f"Loaded **{st.session_state.pdf_name}** · {st.session_state.page_count} pages · {len(st.session_state.chunks)} text chunks")

with st.expander("Document details"):
    st.write(f"**File:** {st.session_state.pdf_name}")
    st.write(f"**Pages:** {st.session_state.page_count}")
    st.write("Only one document is indexed at a time. Upload another PDF to replace it and clear the chat.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("evidence"):
            with st.expander("Evidence used"):
                for item in message["evidence"]:
                    st.markdown(f"**Page {item['page']} · Retrieval score {item['score']:.3f}**")
                    st.write(item["text"])

prompt = st.chat_input("Ask a question about this PDF...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document and preparing an answer..."):
            results = st.session_state.retriever.search(prompt, top_k=top_k)
            usable = [r for r in results if r["score"] >= min_score]

            if not usable:
                answer = ABSTAIN
                evidence = results[:top_k]
                st.markdown(answer)
                st.caption("No passage met the minimum relevance threshold.")
            else:
                # Keep the prompt small for low-memory machines.
                context = "\n\n".join(
                    f"[Page {r['page']}]\n{r['text']}" for r in usable
                )
                try:
                    if not check_ollama():
                        raise RuntimeError("Ollama is not responding. Start Ollama and download the model shown in the sidebar.")
                    answer = ollama_chat(
                        question=prompt,
                        context=context,
                        model=model,
                    )
                    # Guard against empty model responses.
                    if not answer.strip():
                        answer = ABSTAIN
                    evidence = usable
                    st.markdown(answer)
                except Exception as exc:
                    answer = f"**Could not generate an answer.** {exc}"
                    evidence = usable
                    st.error(answer)

            if evidence:
                with st.expander("Evidence used"):
                    for item in evidence:
                        st.markdown(f"**Page {item['page']} · Retrieval score {item['score']:.3f}**")
                        st.write(item["text"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "evidence": evidence,
    })

st.markdown("---")
if st.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.rerun()
