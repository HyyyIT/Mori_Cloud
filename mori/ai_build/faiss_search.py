import json
import torch
import faiss
import numpy as np
import os
from PIL import Image
import open_clip
from .translate import translate_lib  # Giữ lại dịch nếu cần

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class FaissSearch:
    def __init__(self, json_path: str, bin_file: str):
        self.id2img_fps = self._load_json(json_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-SO400M-14-SigLIP-384', device=self.device, pretrained='webli'
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP-384')
        self.index = faiss.read_index(bin_file)  # Load FAISS index từ file

    def _load_json(self, json_path):
        with open(json_path, 'r') as f:
            return {int(k): v for k, v in json.load(f).items()}

    def _extract_features(self, query, mode="text"):
        """Trích xuất đặc trưng cho văn bản hoặc hình ảnh"""
        if mode == "text":
            query = self.tokenizer([query]).to(self.device)
            features = self.model.encode_text(query)
        elif mode == "image":
            query = self.preprocess(Image.open(query).convert("RGB")).to(self.device).unsqueeze(0)
            features = self.model.encode_image(query)

        return features / features.norm(dim=-1, keepdim=True)  # Chuẩn hóa vector

    def _get_frame_info(self, idx_images):
        """Lấy thông tin ảnh từ chỉ số tìm được"""
        return [self.id2img_fps[i] for i in idx_images]

    def search(self, query, mode="text", k=5):
        """Tìm kiếm ảnh tương tự dựa trên văn bản hoặc hình ảnh"""
        if mode == "text":
            query = str(translate_lib(query, to_lang='en'))

        query_features = self._extract_features(query, mode).cpu().detach().numpy().astype(np.float32)
        scores, idx_images = self.index.search(query_features, k=k)

        results = self._get_frame_info(idx_images.flatten())
        return list(zip(results, scores.flatten()))
 
if __name__ == "__main__":
    searcher = FaissSearch("media/id2img_fps.json", "media/faiss_index.bin")

    # Tìm kiếm bằng văn bản
    text_query = "Người dẫn chương trình mặc áo xanh đứng trước nền thành phố lúc hoàng hôn."
    results = searcher.search(text_query, mode="text", k=10)
    print("\n🔍 **Kết quả tìm kiếm văn bản**:")
    for i, (img, score) in enumerate(results):
        print(f"{i+1}. {img} (score: {score:.4f})")
 
    img_query = "D:/Images/sample.jpg"
    results = searcher.search(img_query, mode="image", k=10)
    print("\n🖼️ **Kết quả tìm kiếm hình ảnh**:")
    for i, (img, score) in enumerate(results):
        print(f"{i+1}. {img} (score: {score:.4f})")
