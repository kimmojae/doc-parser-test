import os
import re
import subprocess
import tempfile
import time
import uuid

import gradio as gr
import base64
import openpyxl
import pandas as pd
import pymupdf4llm
from google import genai
from PIL import Image
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from markitdown import MarkItDown

DOCLING_DEVICES = ["auto", "cpu", "cuda", "mps"]

LIBREOFFICE_CONTAINER = os.environ.get("LIBREOFFICE_CONTAINER", "libreoffice-server")


# ──────────────────────────────
# VLM (Gemini) helpers
# ──────────────────────────────


def describe_image_gemini(pil_image, api_key, prompt=None):
    client = genai.Client(api_key=api_key)
    if not prompt:
        prompt = "이 이미지의 내용을 2-3문장으로 설명하세요."
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[pil_image, prompt],
    )
    return response.text


def _img_to_data_uri(img_path):
    """이미지 파일을 base64 data URI로 변환."""
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(img_path)[1].lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    return f"data:{mime};base64,{data}"


def _embed_images_as_base64(md):
    """마크다운 내 ![...](로컬경로) 패턴의 이미지를 base64 data URI로 치환."""
    pattern = r"(!\[([^\]]*)\]\(([^)]+)\))"

    def _replace(match):
        full, alt, img_path = match.group(1), match.group(2), match.group(3)
        if img_path.startswith(("data:", "http://", "https://")):
            return full
        try:
            data_uri = _img_to_data_uri(img_path)
            return f"![{alt}]({data_uri})"
        except Exception:
            return full

    return re.sub(pattern, _replace, md)


def _add_vlm_captions(md, vlm_fn):
    """마크다운 내 ![...](path) 패턴을 찾아 VLM 캡션 추가."""
    pattern = r"(!\[[^\]]*\]\(([^)]+)\))"

    def _replace(match):
        original = match.group(1)
        img_path = match.group(2)
        if img_path.startswith(("data:", "http://", "https://")):
            return original
        try:
            img = Image.open(img_path)
            desc = vlm_fn(img)
            return f"{original}\n*{desc}*"
        except Exception:
            return original

    return re.sub(pattern, _replace, md)


# ──────────────────────────────
# Common: LibreOffice PDF conversion
# ──────────────────────────────


def _convert_to_pdf_with_libreoffice(file_path: str) -> str:
    container = LIBREOFFICE_CONTAINER
    ext = os.path.splitext(file_path)[1]
    safe_name = uuid.uuid4().hex + ext
    safe_pdf = uuid.uuid4().hex + ".pdf"

    # 파일을 컨테이너로 복사 (ASCII-safe 이름 사용)
    subprocess.run(
        ["docker", "cp", file_path, f"{container}:/data/{safe_name}"], check=True
    )
    subprocess.run(
        ["docker", "exec", container, "chmod", "644", f"/data/{safe_name}"], check=True
    )

    # 컨테이너 내에서 LibreOffice 변환
    subprocess.run(
        [
            "docker",
            "exec",
            container,
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            "/data",
            f"/data/{safe_name}",
        ],
        check=True,
    )

    # 변환된 PDF를 로컬로 복사
    converted_name = os.path.splitext(safe_name)[0] + ".pdf"
    tmp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmp_dir, safe_pdf)
    subprocess.run(
        ["docker", "cp", f"{container}:/data/{converted_name}", pdf_path], check=True
    )

    # 컨테이너 내 임시 파일 정리
    subprocess.run(
        [
            "docker",
            "exec",
            container,
            "rm",
            "-f",
            f"/data/{safe_name}",
            f"/data/{converted_name}",
        ],
        check=True,
    )

    return pdf_path


# ──────────────────────────────
# PDF Parser functions
# ──────────────────────────────


def parse_pdf_pymupdf4llm(file_path: str, extract_images=False, vlm_fn=None) -> str:
    if not extract_images:
        return pymupdf4llm.to_markdown(file_path)

    doc_id = uuid.uuid4().hex[:8]
    images_dir = os.path.abspath(os.path.join("output", doc_id, "images"))
    os.makedirs(images_dir, exist_ok=True)
    md = pymupdf4llm.to_markdown(
        file_path, write_images=True, image_path=images_dir, dpi=150
    )

    if vlm_fn:
        md = _add_vlm_captions(md, vlm_fn)
    return md


def parse_pdf_docling(
    file_path: str, device: str = "auto", extract_images=False, vlm_fn=None
) -> str:
    accel = AcceleratorOptions(device=device)
    pipeline_options = PdfPipelineOptions(
        accelerator_options=accel,
        generate_picture_images=extract_images,
        images_scale=2.0 if extract_images else 1.0,
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(file_path)
    document = result.document
    md = document.export_to_markdown()

    if extract_images and document.pictures:
        doc_id = uuid.uuid4().hex[:8]
        images_dir = os.path.abspath(os.path.join("output", doc_id, "images"))
        os.makedirs(images_dir, exist_ok=True)
        for idx, pic in enumerate(document.pictures):
            if not pic.image:
                continue
            img_path = os.path.join(images_dir, f"image_{idx}.png")
            pic.image.pil_image.save(img_path, format="PNG")
            desc = vlm_fn(pic.image.pil_image) if vlm_fn else ""
            if desc:
                caption = f"\n![Image {idx}]({img_path})\n*{desc}*\n"
            else:
                caption = f"\n![Image {idx}]({img_path})\n"
            md = md.replace("<!-- image -->", caption, 1)
    return md


PDF_PARSERS = {
    "pymupdf4llm": parse_pdf_pymupdf4llm,
    "docling": parse_pdf_docling,
}

# ──────────────────────────────
# Excel Parser functions
# ──────────────────────────────


def parse_excel_openpyxl(file_path: str) -> str:
    wb = openpyxl.load_workbook(file_path)
    ws = wb.active
    data = [[cell.value for cell in row] for row in ws.iter_rows()]
    if data:
        df = pd.DataFrame(data[1:], columns=data[0])
    else:
        df = pd.DataFrame()
    return df.to_markdown(index=False)


def parse_excel_pandas(file_path: str) -> str:
    df = pd.read_excel(file_path)
    return df.to_markdown(index=False)


def parse_excel_markitdown(file_path: str) -> str:
    md = MarkItDown()
    result = md.convert(file_path)
    return result.text_content


def parse_excel_docling(file_path: str, device: str = "auto") -> str:
    accel = AcceleratorOptions(device=device)
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()


EXCEL_PARSERS = {
    "openpyxl": parse_excel_openpyxl,
    "pandas": parse_excel_pandas,
    "markitdown": parse_excel_markitdown,
    "docling": parse_excel_docling,
}

# ──────────────────────────────
# DOCX Parser functions
# ──────────────────────────────


def parse_docx_markitdown(file_path: str) -> str:
    md = MarkItDown()
    result = md.convert(file_path)
    return result.text_content


def parse_docx_docling(file_path: str, device: str = "auto") -> str:
    accel = AcceleratorOptions(device=device)
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()


DOCX_PARSERS = {
    "markitdown": parse_docx_markitdown,
    "docling": parse_docx_docling,
}

# ──────────────────────────────
# PPTX Parser functions
# ──────────────────────────────


def parse_pptx_markitdown(file_path: str) -> str:
    md = MarkItDown()
    result = md.convert(file_path)
    return result.text_content


def parse_pptx_docling(file_path: str, device: str = "auto") -> str:
    accel = AcceleratorOptions(device=device)
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()


PPTX_PARSERS = {
    "markitdown": parse_pptx_markitdown,
    "docling": parse_pptx_docling,
}

# ──────────────────────────────
# HWP Parser functions
# ──────────────────────────────


def parse_hwp_pymupdf4llm(file_path: str) -> str:
    pdf_path = _convert_to_pdf_with_libreoffice(file_path)
    return pymupdf4llm.to_markdown(pdf_path)


def parse_hwp_docling(file_path: str, device: str = "auto") -> str:
    pdf_path = _convert_to_pdf_with_libreoffice(file_path)
    accel = AcceleratorOptions(device=device)
    pipeline_options = PdfPipelineOptions(accelerator_options=accel)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()


HWP_PARSERS = {
    "pymupdf4llm": parse_hwp_pymupdf4llm,
    "docling": parse_hwp_docling,
}

# ──────────────────────────────
# PDF conversion logic
# ──────────────────────────────


def convert_pdf(
    file, selected_parsers, docling_device, extract_images, gemini_api_key, enable_vlm
):
    if file is None:
        raise gr.Error("PDF 파일을 업로드해주세요.")
    if not selected_parsers:
        raise gr.Error("파서를 하나 이상 선택해주세요.")

    vlm_fn = None
    if extract_images and enable_vlm and gemini_api_key:
        vlm_fn = lambda img: describe_image_gemini(img, gemini_api_key)

    results = {}
    for name in selected_parsers:
        if name in PDF_PARSERS:
            try:
                start = time.time()
                if name == "docling":
                    md = PDF_PARSERS[name](
                        file.name,
                        device=docling_device,
                        extract_images=extract_images,
                        vlm_fn=vlm_fn,
                    )
                else:
                    md = PDF_PARSERS[name](
                        file.name,
                        extract_images=extract_images,
                        vlm_fn=vlm_fn,
                    )
                elapsed = time.time() - start
                results[name] = {"markdown": md, "time": elapsed}
            except Exception as e:
                results[name] = {"markdown": f"오류 발생: {e}", "time": 0}

    outputs = []
    for parser_name in PDF_PARSERS:
        if parser_name in results:
            r = results[parser_name]
            outputs.append(r["markdown"])  # Code 탭: 경로 그대로
            outputs.append(_embed_images_as_base64(r["markdown"]))  # Rendered 탭: base64
            outputs.append(f"⏱ {r['time']:.2f}초")
        else:
            outputs.append("")
            outputs.append("")
            outputs.append("")

    return outputs


# ──────────────────────────────
# Excel conversion logic
# ──────────────────────────────


def convert_excel(file, selected_parsers, docling_device):
    if file is None:
        raise gr.Error("Excel 파일을 업로드해주세요.")
    if not selected_parsers:
        raise gr.Error("파서를 하나 이상 선택해주세요.")

    results = {}
    for name in selected_parsers:
        if name in EXCEL_PARSERS:
            try:
                start = time.time()
                if name == "docling":
                    md = EXCEL_PARSERS[name](file.name, device=docling_device)
                else:
                    md = EXCEL_PARSERS[name](file.name)
                elapsed = time.time() - start
                results[name] = {"markdown": md, "time": elapsed}
            except Exception as e:
                results[name] = {"markdown": f"오류 발생: {e}", "time": 0}

    outputs = []
    for parser_name in EXCEL_PARSERS:
        if parser_name in results:
            r = results[parser_name]
            outputs.append(r["markdown"])
            outputs.append(r["markdown"])
            outputs.append(f"⏱ {r['time']:.2f}초")
        else:
            outputs.append("")
            outputs.append("")
            outputs.append("")

    return outputs


def _generic_convert(file, selected_parsers, docling_device, parsers_dict, file_label):
    if file is None:
        raise gr.Error(f"{file_label} 파일을 업로드해주세요.")
    if not selected_parsers:
        raise gr.Error("파서를 하나 이상 선택해주세요.")

    results = {}
    for name in selected_parsers:
        if name in parsers_dict:
            try:
                start = time.time()
                if name == "docling":
                    md = parsers_dict[name](file.name, device=docling_device)
                else:
                    md = parsers_dict[name](file.name)
                elapsed = time.time() - start
                results[name] = {"markdown": md, "time": elapsed}
            except Exception as e:
                results[name] = {"markdown": f"오류 발생: {e}", "time": 0}

    outputs = []
    for parser_name in parsers_dict:
        if parser_name in results:
            r = results[parser_name]
            outputs.append(r["markdown"])
            outputs.append(r["markdown"])
            outputs.append(f"⏱ {r['time']:.2f}초")
        else:
            outputs.append("")
            outputs.append("")
            outputs.append("")

    return outputs


def convert_docx(file, selected_parsers, docling_device):
    return _generic_convert(
        file, selected_parsers, docling_device, DOCX_PARSERS, "DOCX"
    )


def convert_pptx(file, selected_parsers, docling_device):
    return _generic_convert(
        file, selected_parsers, docling_device, PPTX_PARSERS, "PPTX"
    )


def convert_hwp(file, selected_parsers, docling_device):
    return _generic_convert(file, selected_parsers, docling_device, HWP_PARSERS, "HWP")


def convert_file_to_pdf(file):
    if file is None:
        raise gr.Error("파일을 업로드해주세요.")
    try:
        pdf_path = _convert_to_pdf_with_libreoffice(file.name)
        with open(pdf_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        viewer_html = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="800px"></iframe>'
        return pdf_path, viewer_html
    except Exception as e:
        raise gr.Error(f"PDF 변환 오류: {e}")


# ──────────────────────────────
# UI
# ──────────────────────────────

with gr.Blocks(title="Document Parser Tester") as demo:
    gr.Markdown("# Document Parser Tester")

    with gr.Tab("PDF"):
        with gr.Row():
            with gr.Column(scale=3):
                pdf_file_input = gr.File(label="PDF 업로드", file_types=[".pdf"])
            with gr.Column(scale=1):
                pdf_parser_selection = gr.CheckboxGroup(
                    choices=list(PDF_PARSERS.keys()),
                    value=list(PDF_PARSERS.keys()),
                    label="파서 선택",
                )
                pdf_docling_device = gr.Dropdown(
                    choices=DOCLING_DEVICES,
                    value="cpu",
                    label="Docling 디바이스",
                )
                pdf_extract_images = gr.Checkbox(
                    label="이미지 추출", value=False
                )
                pdf_gemini_api_key = gr.Textbox(
                    label="Gemini API Key",
                    type="password",
                    value=os.environ.get("GEMINI_API_KEY", ""),
                )
                pdf_enable_vlm = gr.Checkbox(
                    label="VLM 설명 생성", value=False
                )
                pdf_convert_btn = gr.Button("Parsing", variant="primary")

        pdf_tabs = {}
        pdf_outputs = {}

        for parser_name in PDF_PARSERS:
            with gr.Tab(label=parser_name, visible=True) as tab:
                elapsed_label = gr.Markdown("")
                with gr.Tab("Markdown"):
                    code_output = gr.Code(
                        language="markdown", wrap_lines=True, show_label=False
                    )
                with gr.Tab("Rendered"):
                    rendered_output = gr.Markdown(show_label=False)

            pdf_outputs[parser_name] = (code_output, rendered_output, elapsed_label)

        all_pdf_outputs = []
        pdf_code_list = []
        for parser_name in PDF_PARSERS:
            code, rendered, elapsed = pdf_outputs[parser_name]
            all_pdf_outputs.extend([code, rendered, elapsed])
            pdf_code_list.append(code)

        pdf_convert_btn.click(
            fn=convert_pdf,
            inputs=[
                pdf_file_input,
                pdf_parser_selection,
                pdf_docling_device,
                pdf_extract_images,
                pdf_gemini_api_key,
                pdf_enable_vlm,
            ],
            outputs=all_pdf_outputs,
            show_progress="full",
            show_progress_on=pdf_code_list,
        )

    with gr.Tab("DOCX"):
        with gr.Row():
            with gr.Column(scale=3):
                docx_file_input = gr.File(label="DOCX 업로드", file_types=[".docx"])
            with gr.Column(scale=1):
                docx_parser_selection = gr.CheckboxGroup(
                    choices=list(DOCX_PARSERS.keys()),
                    value=list(DOCX_PARSERS.keys()),
                    label="파서 선택",
                )
                docx_docling_device = gr.Dropdown(
                    choices=DOCLING_DEVICES,
                    value="cpu",
                    label="Docling 디바이스",
                )
                docx_convert_btn = gr.Button("Parsing", variant="primary")

        docx_outputs = {}
        for parser_name in DOCX_PARSERS:
            with gr.Tab(label=parser_name, visible=True):
                elapsed_label = gr.Markdown("")
                with gr.Tab("Markdown"):
                    code_output = gr.Code(
                        language="markdown", wrap_lines=True, show_label=False
                    )
                with gr.Tab("Rendered"):
                    rendered_output = gr.Markdown(show_label=False)
            docx_outputs[parser_name] = (code_output, rendered_output, elapsed_label)

        all_docx_outputs = []
        for parser_name in DOCX_PARSERS:
            code, rendered, elapsed = docx_outputs[parser_name]
            all_docx_outputs.extend([code, rendered, elapsed])

        docx_convert_btn.click(
            fn=convert_docx,
            inputs=[docx_file_input, docx_parser_selection, docx_docling_device],
            outputs=all_docx_outputs,
        )

    with gr.Tab("PPTX"):
        with gr.Row():
            with gr.Column(scale=3):
                pptx_file_input = gr.File(label="PPTX 업로드", file_types=[".pptx"])
            with gr.Column(scale=1):
                pptx_parser_selection = gr.CheckboxGroup(
                    choices=list(PPTX_PARSERS.keys()),
                    value=list(PPTX_PARSERS.keys()),
                    label="파서 선택",
                )
                pptx_docling_device = gr.Dropdown(
                    choices=DOCLING_DEVICES,
                    value="cpu",
                    label="Docling 디바이스",
                )
                pptx_convert_btn = gr.Button("Parsing", variant="primary")

        pptx_outputs = {}
        for parser_name in PPTX_PARSERS:
            with gr.Tab(label=parser_name, visible=True):
                elapsed_label = gr.Markdown("")
                with gr.Tab("Markdown"):
                    code_output = gr.Code(
                        language="markdown", wrap_lines=True, show_label=False
                    )
                with gr.Tab("Rendered"):
                    rendered_output = gr.Markdown(show_label=False)
            pptx_outputs[parser_name] = (code_output, rendered_output, elapsed_label)

        all_pptx_outputs = []
        for parser_name in PPTX_PARSERS:
            code, rendered, elapsed = pptx_outputs[parser_name]
            all_pptx_outputs.extend([code, rendered, elapsed])

        pptx_convert_btn.click(
            fn=convert_pptx,
            inputs=[pptx_file_input, pptx_parser_selection, pptx_docling_device],
            outputs=all_pptx_outputs,
        )

    with gr.Tab("Excel"):
        with gr.Row():
            with gr.Column(scale=3):
                excel_file_input = gr.File(
                    label="Excel 업로드", file_types=[".xlsx", ".xls"]
                )
            with gr.Column(scale=1):
                excel_parser_selection = gr.CheckboxGroup(
                    choices=list(EXCEL_PARSERS.keys()),
                    value=list(EXCEL_PARSERS.keys()),
                    label="파서 선택",
                )
                excel_docling_device = gr.Dropdown(
                    choices=DOCLING_DEVICES,
                    value="cpu",
                    label="Docling 디바이스",
                )
                excel_convert_btn = gr.Button("Parsing", variant="primary")

        excel_outputs = {}
        for parser_name in EXCEL_PARSERS:
            with gr.Tab(label=parser_name, visible=True):
                elapsed_label = gr.Markdown("")
                with gr.Tab("Markdown"):
                    code_output = gr.Code(
                        language="markdown", wrap_lines=True, show_label=False
                    )
                with gr.Tab("Rendered"):
                    rendered_output = gr.Markdown(show_label=False)
            excel_outputs[parser_name] = (code_output, rendered_output, elapsed_label)

        all_excel_outputs = []
        for parser_name in EXCEL_PARSERS:
            code, rendered, elapsed = excel_outputs[parser_name]
            all_excel_outputs.extend([code, rendered, elapsed])

        excel_convert_btn.click(
            fn=convert_excel,
            inputs=[excel_file_input, excel_parser_selection, excel_docling_device],
            outputs=all_excel_outputs,
        )

    with gr.Tab("HWP"):
        with gr.Row():
            with gr.Column(scale=3):
                hwp_file_input = gr.File(
                    label="HWP 업로드", file_types=[".hwp", ".hwpx"]
                )
            with gr.Column(scale=1):
                hwp_parser_selection = gr.CheckboxGroup(
                    choices=list(HWP_PARSERS.keys()),
                    value=list(HWP_PARSERS.keys()),
                    label="파서 선택",
                )
                hwp_docling_device = gr.Dropdown(
                    choices=DOCLING_DEVICES,
                    value="cpu",
                    label="Docling 디바이스",
                )
                hwp_convert_btn = gr.Button("Parsing", variant="primary")

        hwp_outputs = {}
        for parser_name in HWP_PARSERS:
            with gr.Tab(label=parser_name, visible=True):
                elapsed_label = gr.Markdown("")
                with gr.Tab("Markdown"):
                    code_output = gr.Code(
                        language="markdown", wrap_lines=True, show_label=False
                    )
                with gr.Tab("Rendered"):
                    rendered_output = gr.Markdown(show_label=False)
            hwp_outputs[parser_name] = (code_output, rendered_output, elapsed_label)

        all_hwp_outputs = []
        for parser_name in HWP_PARSERS:
            code, rendered, elapsed = hwp_outputs[parser_name]
            all_hwp_outputs.extend([code, rendered, elapsed])

        hwp_convert_btn.click(
            fn=convert_hwp,
            inputs=[hwp_file_input, hwp_parser_selection, hwp_docling_device],
            outputs=all_hwp_outputs,
        )

    with gr.Tab("PDF Convert"):
        with gr.Row():
            with gr.Column(scale=3):
                pdf_conv_file_input = gr.File(label="파일 업로드")
            with gr.Column(scale=1):
                pdf_conv_btn = gr.Button("Convert to PDF", variant="primary")
        pdf_conv_output = gr.File(label="변환된 PDF 다운로드")
        pdf_conv_viewer = gr.HTML(label="PDF 미리보기")

        pdf_conv_btn.click(
            fn=convert_file_to_pdf,
            inputs=[pdf_conv_file_input],
            outputs=[pdf_conv_output, pdf_conv_viewer],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", allowed_paths=["output"])
