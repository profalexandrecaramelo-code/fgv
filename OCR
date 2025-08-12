import io
import time
from typing import List, Tuple

import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import streamlit as st

# EasyOCR é "pesado" no primeiro load; vamos cachear o Reader
@st.cache_resource(show_spinner=False)
def get_reader(langs: Tuple[str, ...]):
    import easyocr
    return easyocr.Reader(list(langs), gpu=False)  # Streamlit Cloud geralmente sem GPU

def pdf_pages_to_images(file_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
    """
    Converte um PDF em uma lista de imagens PIL (uma por página), sem dependências externas.
    """
    images = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            # Renderiza a página em raster
            zoom = dpi / 72  # 72 dpi é o padrão do PDF
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images

def imagefile_to_pil(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

def run_ocr_on_images(images: List[Image.Image], langs: Tuple[str, ...], detail: int = 0):
    """
    Roda EasyOCR nas imagens. detail=0 -> retorna só texto; 1/2 retornam caixas e confianças.
    """
    reader = get_reader(langs)
    results = []
    for idx, img in enumerate(images, start=1):
        # EasyOCR espera numpy array
        arr = np.array(img)
        res = reader.readtext(arr, detail=detail, paragraph=True)
        results.append(res)
    return results

def stitch_text(results) -> str:
    """
    Junta o texto página a página. O formato do resultado depende do 'detail'.
    - detail=0 => lista de strings por página
    """
    pages_text = []
    for p, page_res in enumerate(results, start=1):
        if isinstance(page_res, list) and page_res and isinstance(page_res[0], str):
            text = "\n".join(page_res)
        else:
            # Caso detail != 0, extrair apenas a string do tuple (bbox, text, conf)
            lines = []
            for item in page_res:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    lines.append(item[1])
            text = "\n".join(lines)
        pages_text.append(f"=== Página {p} ===\n{text}".strip())
    return "\n\n".join(pages_text).strip()

def make_docx(text: str) -> bytes:
    from docx import Document
    doc = Document()
    for block in text.split("\n\n"):
        doc.add_paragraph(block)
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.read()

# ---------------- UI ----------------
st.set_page_config(page_title="OCR de PDF (pt-BR)", page_icon="📝", layout="wide")
st.title("📝 OCR de PDF e Imagens (pt-BR)")

st.write(
    "Faça upload de um **PDF escaneado** ou **imagem** (JPG/PNG). "
    "O app converte as páginas em imagens e aplica **EasyOCR** para extrair o texto."
)

with st.sidebar:
    st.header("Configurações")
    # Idiomas do OCR (padrão: português + inglês ajuda na acurácia mista)
    lang_options = {
        "Português (pt)": "pt",
        "Inglês (en)": "en",
        "Espanhol (es)": "es",
        "Francês (fr)": "fr",
    }
    selected = st.multiselect(
        "Idiomas para o OCR (quanto menos, mais rápido):",
        list(lang_options.keys()),
        default=["Português (pt)", "Inglês (en)"]
    )
    langs = tuple(lang_options[k] for k in selected) or ("pt",)
    dpi = st.slider("DPI para renderizar PDF (qualidade x velocidade)", 120, 300, 200, 10)
    detail = st.selectbox("Detalhe do OCR", [0, 1], index=0, help="0 = só texto; 1 = caixas + texto (mais lento)")
    show_previews = st.checkbox("Mostrar prévias das páginas", value=False)

uploaded = st.file_uploader(
    "Envie um PDF ou imagem",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=False
)

if uploaded:
    file_bytes = uploaded.read()
    suffix = uploaded.name.lower().split(".")[-1]

    # 1) Converter para imagens
    with st.spinner("Convertendo para imagens..."):
        t0 = time.time()
        if suffix == "pdf":
            images = pdf_pages_to_images(file_bytes, dpi=dpi)
        else:
            images = [imagefile_to_pil(file_bytes)]
        conv_time = time.time() - t0

    st.success(f"Conversão concluída em {conv_time:.1f}s • {len(images)} página(s)/imagem(ns)")

    if show_previews:
        st.subheader("Pré-visualização")
        cols = st.columns(2)
        for i, img in enumerate(images):
            cols[i % 2].image(img, caption=f"Página {i+1}", use_container_width=True)

    # 2) Rodar OCR
    if st.button("🔎 Rodar OCR agora", type="primary"):
        with st.spinner("Executando OCR (isso pode levar alguns segundos na primeira vez)..."):
            t1 = time.time()
            results = run_ocr_on_images(images, langs=langs, detail=detail)
            ocr_time = time.time() - t1

        # 3) Extrair texto
        extracted_text = stitch_text(results) if results else ""
        st.success(f"OCR finalizado em {ocr_time:.1f}s")
        st.subheader("Texto extraído")
        if extracted_text.strip():
            st.text_area("Resultado", extracted_text, height=400)

            # Downloads
            txt_bytes = extracted_text.encode("utf-8", errors="ignore")
            st.download_button("⬇️ Baixar como .txt", data=txt_bytes, file_name="ocr_resultado.txt", mime="text/plain")

            docx_bytes = make_docx(extracted_text)
            st.download_button("⬇️ Baixar como .docx", data=docx_bytes, file_name="ocr_resultado.docx",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        else:
            st.warning("Não foi possível extrair texto. Verifique a qualidade da imagem/PDF e os idiomas selecionados.")

else:
    st.info("Envie um arquivo para começar. Dica: quanto melhor o contraste e a nitidez, melhor o OCR.")
