from pathlib import Path
import shutil

src = Path(r"D:\idea\project\NEU-CLS\NEU-CLS")
dst = Path(r"D:\idea\project\NEU-CLS")

classes = ["Cr", "In", "Pa", "PS", "RS", "Sc"]

for img in src.glob("*.bmp"):
    label = img.stem.split("_")[0]

    if label in classes:
        (dst / label).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst / label / img.name)