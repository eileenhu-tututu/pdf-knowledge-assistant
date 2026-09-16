import fitz

from upload_service import extract_page_chunks


PDF_PATH = (
    "documents/"
    "solax-x3-aelio-datasheet-en.pdf"
)


pdf = fitz.open(
    PDF_PATH
)


for page_number, page in enumerate(
    pdf,
    start=1
):

    chunks = extract_page_chunks(
        page=page,
        filename=PDF_PATH,
        page_number=page_number
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PAGE:",
        page_number
    )

    print(
        "=" * 80
    )

    for chunk in (
        chunks[:20]
    ):

        print(
            chunk["content"]
        )

        print(
            "-" * 50
        )