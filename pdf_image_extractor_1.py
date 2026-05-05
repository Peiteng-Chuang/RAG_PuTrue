import sys
from pathlib import Path
from pypdf import PdfReader

def extract_images(pdf_path: str, output_dir: str = "extracted_images"):
    output = Path(output_dir)
    output.mkdir(exist_ok=True)

    reader = PdfReader(pdf_path)
    total = 0

    for page_num, page in enumerate(reader.pages, start=1):
        image_count = 0

        for img_obj in page.images:
            image_count += 1
            filename = f"{page_num}_{image_count}{Path(img_obj.name).suffix or '.png'}"
            filepath = output / filename

            with open(filepath, "wb") as f:
                f.write(img_obj.data)

            print(f"  saved: {filename}")
            total += 1

        if image_count == 0:
            print(f"  page {page_num}: no images")

    print(f"\n完成，共提取 {total} 張圖片 → {output.resolve()}")

if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "C:/project_file/RAG_PuTrue/test2.pdf"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./pdf_extractor_images"
    extract_images(pdf_path, output_dir)