import os
import torch
import open_clip
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm

class FaissImageIndexer:
    def __init__(self, model_name='ViT-SO400M-14-SigLIP-384', dataset_name='webli', index_path="faiss_clip.bin", feature_dim=512):
        """
        Khởi tạo class FaissImageIndexer để trích xuất đặc trưng ảnh và lưu vào FAISS index.
        
        Args:
            model_name (str): Tên model OpenCLIP.
            dataset_name (str): Tên dataset pretrain.
            index_path (str): Đường dẫn file FAISS index.
            feature_dim (int): Kích thước vector đặc trưng.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.index_path = index_path
        self.feature_dim = feature_dim

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, device=self.device, pretrained=dataset_name
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.index = self.load_faiss_index()

    def extract_image_features(self, image_path):
        """Trích xuất đặc trưng của một ảnh."""
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)  
        return image_features.cpu().detach().numpy().squeeze().astype(np.float32)

    def create_faiss_index(self):
        """Tạo FAISS index mới (Inner Product - IP)."""
        return faiss.IndexFlatIP(self.feature_dim)

    def save_faiss_index(self):
        """Lưu FAISS index vào file."""
        faiss.write_index(self.index, self.index_path)
        print(f"🔹 FAISS index đã lưu vào: {self.index_path}")

    def load_faiss_index(self):
        """Tải FAISS index từ file, nếu không có thì tạo mới."""
        if os.path.exists(self.index_path):
            index = faiss.read_index(self.index_path)
            print(f"✅ FAISS index loaded từ {self.index_path}")
        else:
            print(f"⚠️ Không tìm thấy {self.index_path}, tạo FAISS index mới.")
            index = self.create_faiss_index()
        return index

    def add_images(self, image_paths):
        """Thêm danh sách ảnh vào FAISS index."""
        for img_path in tqdm(image_paths, desc="🔄 Trích xuất & Thêm đặc trưng vào FAISS"):
            try:
                feature = self.extract_image_features(img_path).reshape(1, -1)
                self.index.add(feature)
            except Exception as e:
                print(f"❌ Lỗi khi xử lý ảnh {img_path}: {e}")

        self.save_faiss_index()
 
if __name__ == "__main__":
    indexer = FaissImageIndexer(index_path="faiss_clip.bin")
 
    image_list = [r"D:\University\N3_HK2\PTUD\Mori\anh.jpg"]  
    indexer.add_images(image_list)
 
    image_list = [r"D:\University\N3_HK2\PTUD\Mori\image.png"]
    indexer.add_images(image_list)
