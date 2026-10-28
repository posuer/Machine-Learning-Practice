"""Local integration demo, single worker. Not a hardened production service."""
import argparse
import io
import threading

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


def create_app(extractor):
    app = FastAPI(title="Receipt extraction mock onsite")
    lock = threading.Lock()  # Serialize GPU work in this minimal demo.

    @app.get("/health")
    def health():
        return {"ready": True}

    @app.post("/extract")
    def extract(file: UploadFile = File(...)):
        data = file.file.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise HTTPException(413, "Maximum uploaded file size is 8 MiB")
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > 40_000_000:
                    raise HTTPException(413, "Maximum decoded image size is 40 megapixels")
                image.load()
                image = image.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(400, "Invalid image")
        with lock:
            return extractor.predict(image)

    return app


def main():
    import uvicorn
    from .predict import Extractor
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/baseline.json")
    p.add_argument("--adapter")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run(create_app(Extractor(args.config, args.adapter)), host="127.0.0.1", port=args.port, workers=1)


if __name__ == '__main__':
    main()
