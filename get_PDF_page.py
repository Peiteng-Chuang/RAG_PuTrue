from pypdf import PdfReader, PdfWriter

reader = PdfReader("C:/Users/Peiteng.Chuang/Desktop/璞真RAG/本因坊結案報告(提審)95.08.01.pdf")
writer = PdfWriter()

writer.add_page(reader.pages[68])

with open("page_69.pdf", "wb") as f:
    writer.write(f)