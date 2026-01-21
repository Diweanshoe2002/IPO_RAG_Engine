from pathlib import Path
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from llama_index.core.schema import Document

def docling_chunk_pdf(pdf_path, tokenizer, use_gpu=True):
    accelerator = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CUDA if use_gpu else AcceleratorDevice.CPU
    )

    pipeline = PdfPipelineOptions()
    pipeline.accelerator_options = accelerator
    pipeline.do_ocr = True
    pipeline.do_table_structure = True
    pipeline.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline)}
    )

    result = converter.convert(Path(pdf_path))
    doc = result.document

    chunker = HybridChunker(tokenizer=tokenizer)
    chunks = chunker.chunk(doc)

    return [
        Document(
            text=c.text,
            metadata={"source": pdf_path, **(c.meta if hasattr(c, "meta") else {})}
        )
        for c in chunks
    ]
