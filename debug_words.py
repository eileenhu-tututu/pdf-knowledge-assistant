import fitz

pdf = fitz.open(
    "documents/solax-x3-aelio-datasheet-en.pdf"
)

page = pdf[0]

words = page.get_text("words")

for word in words:
    x0, y0, x1, y1, text, *_ = word

    if (
        "AELIO" in text.upper()
        or "49.9" in text
        or "50K" in text.upper()
        or "60K" in text.upper()
        or "61K" in text.upper()
    ):
        print(
            f"x={x0:.1f} "
            f"y={y0:.1f} "
            f"text={text}"
        )