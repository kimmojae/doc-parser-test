import time

import gradio as gr
import openpyxl
import pandas as pd
import pymupdf4llm
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from markitdown import MarkItDown

DOCLING_DEVICES = ["auto", "cpu", "cuda", "mps"]

# ──────────────────────────────
# PDF Parser functions
# ──────────────────────────────

def parse_pdf_pymupdf4llm(file_path: str) -> str:
    return pymupdf4llm.to_markdown(file_path)


def parse_pdf_docling(file_path: str, device: str = "auto") -> str:
    accel = AcceleratorOptions(device=device)
    pipeline_options = PdfPipelineOptions(accelerator_options=accel)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(file_path)
    return result.document.export_to_markdown()


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
# PDF conversion logic
# ──────────────────────────────

def convert_pdf(file, selected_parsers, docling_device):
    if file is None:
        raise gr.Error("PDF 파일을 업로드해주세요.")
    if not selected_parsers:
        raise gr.Error("파서를 하나 이상 선택해주세요.")

    results = {}
    for name in selected_parsers:
        if name in PDF_PARSERS:
            try:
                start = time.time()
                if name == "docling":
                    md = PDF_PARSERS[name](file.name, device=docling_device)
                else:
                    md = PDF_PARSERS[name](file.name)
                elapsed = time.time() - start
                results[name] = {"markdown": md, "time": elapsed}
            except Exception as e:
                results[name] = {"markdown": f"오류 발생: {e}", "time": 0}

    outputs = []
    for parser_name in PDF_PARSERS:
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
                pdf_convert_btn = gr.Button("변환 실행", variant="primary")

        pdf_tabs = {}
        pdf_outputs = {}

        for parser_name in PDF_PARSERS:
            with gr.Tab(label=parser_name, visible=True) as tab:
                elapsed_label = gr.Markdown("")
                with gr.Tab("Markdown"):
                    code_output = gr.Code(language="markdown", wrap_lines=True, show_label=False)
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
            inputs=[pdf_file_input, pdf_parser_selection, pdf_docling_device],
            outputs=all_pdf_outputs,
            show_progress="full",
            show_progress_on=pdf_code_list,
        )

    with gr.Tab("Excel"):
        with gr.Row():
            with gr.Column(scale=3):
                excel_file_input = gr.File(label="Excel 업로드", file_types=[".xlsx", ".xls"])
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
                excel_convert_btn = gr.Button("변환 실행", variant="primary")

        excel_outputs = {}

        for parser_name in EXCEL_PARSERS:
            with gr.Tab(label=parser_name, visible=True):
                elapsed_label = gr.Markdown("")
                with gr.Tab("Markdown"):
                    code_output = gr.Code(language="markdown", wrap_lines=True, show_label=False)
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

if __name__ == "__main__":
    demo.launch()
