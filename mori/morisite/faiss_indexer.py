import os
import torch
import faiss
import numpy as np
from PIL import Image
import open_clip
from django.conf import settings
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from .models import Photo
import time
import logging
from tqdm import tqdm
from django.utils import timezone

class FaissImageIndexer:
    def __init__(self, user=None, model_name='clip-ViT-B-32', feature_dim=512, chunk_size=256):
        self.device = "cpu"
        self.feature_dim = feature_dim
        self.chunk_size = chunk_size
        self.user = user
        self.index = self.load_faiss_index()
        self.model = SentenceTransformer(model_name, device=self.device)

    @property
    def faiss_index_path(self):
        if self.user:
            file_name = f"faiss_index_user_{self.user.id_user}.bin"
        else:
            file_name = "faiss_index_global.bin"
        return os.path.join(settings.MEDIA_ROOT, file_name)

    def create_faiss_index(self):
        """Tạo FAISS Index mới"""
        print(f"⚠️ Tạo mới FAISS Index tại {self.faiss_index_path}")
        index = faiss.IndexFlatIP(self.feature_dim)
        self.save_faiss_index(index)  
        time.sleep(0.2)  
        return index

    def load_faiss_index(self):
        """Tải hoặc tạo FAISS index riêng cho từng user hoặc global"""
        try:
            if os.path.exists(self.faiss_index_path):
                print(f"✅ FAISS index loaded từ {self.faiss_index_path}")
                return faiss.read_index(self.faiss_index_path)
            else:
                print(f"⚠️ Không tìm thấy {self.faiss_index_path}, tạo FAISS index mới.")
                return self.create_faiss_index()
        except Exception as e:
            print(f"❌ Lỗi khi tải FAISS index: {e}")
            return self.create_faiss_index()

    def save_faiss_index(self, index):
        """Lưu FAISS index vào file"""
        try:
            os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
            faiss.write_index(index, self.faiss_index_path)
            print(f"✅ FAISS index đã được lưu vào {self.faiss_index_path}")
        except Exception as e:
            print(f"❌ Lỗi khi lưu FAISS index: {e}")

    def load_and_preprocess(self, path):
        try:
            if not os.path.exists(path):
                print(f"❌ Ảnh không tồn tại: {path}")
                return None
            image = Image.open(path).convert("RGB")
            return image
        except Exception as e:
            print(f"❌ Lỗi đọc ảnh {path}: {e}")
            return None

    def extract_image_features_chunk(self, image_paths):
        with ThreadPoolExecutor() as executor:
            images = list(executor.map(self.load_and_preprocess, image_paths))

        valid_images = [img for img in images if img is not None]
        if not valid_images:
            raise ValueError("❌ Không có ảnh hợp lệ trong chunk!")

        with torch.no_grad():
            features = self.model.encode(valid_images, batch_size=len(valid_images), convert_to_numpy=True)

        features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        return features.astype(np.float32)

    def add_photo_to_faiss(self, photo):
        try:
            if not os.path.exists(photo.photo.path):
                print(f"❌ Ảnh không tồn tại: {photo.photo.path}")
                return False

            feature_vector = self.extract_image_features_chunk([photo.photo.path])
            if feature_vector is None or len(feature_vector) == 0:
                print(f"⚠️ Ảnh {photo.id_photo} không thể trích xuất đặc trưng.")
                return False

            if feature_vector.shape[1] != self.feature_dim:
                print(f"❌ Kích thước vector của ảnh {photo.id_photo} ({feature_vector.shape[1]}) "
                      f"không khớp với FAISS ({self.feature_dim}) → Bỏ qua ảnh này.")
                return False

            if self.index is None:
                print(f"⚠️ FAISS chưa tồn tại → Tạo FAISS mới")
                self.index = faiss.IndexFlatIP(self.feature_dim)
            
            faiss_id = self.index.ntotal
            self.index.add(feature_vector)

            if self.user is None:
                print(f"FAISS index global: {faiss_id}")
                photo.faiss_id_public = faiss_id
            else:
                print(f"FAISS index user {self.user.id_user}: {faiss_id}")
                photo.faiss_id = faiss_id
            photo.save()

            if photo.faiss_id is None and self.user is not None:
                raise ValueError(f"❌ Không thể cập nhật faiss_id cho ảnh {photo.id_photo}")

            self.save_faiss_index(self.index)
            print(f"✅ Ảnh {photo.id_photo} đã được thêm vào FAISS với ID: {faiss_id}")
            return faiss_id
        except Exception as e:
            print(f"❌ Lỗi khi thêm ảnh vào FAISS: {e}")
            return False
        
    def add_images(self, photos):
        image_paths = [photo.photo.path for photo in photos]
        print(f"🔄 Bắt đầu trích xuất & thêm vào FAISS index với chunk size = {self.chunk_size}")
        for i in tqdm(range(0, len(image_paths), self.chunk_size), desc="🔄 Trích xuất & Thêm chunk"):
            chunk_paths = image_paths[i:i + self.chunk_size]
            chunk_photos = photos[i:i + self.chunk_size]
            try:
                features = self.extract_image_features_chunk(chunk_paths)
                start_id = self.index.ntotal
                self.index.add(features)
                for j, photo in enumerate(chunk_photos):
                    faiss_id = start_id + j
                    if self.user is None:
                        photo.faiss_id_public = faiss_id
                    else:
                        photo.faiss_id = faiss_id
                    photo.save()
                    print(f"✅ Ảnh {photo.id_photo} đã được thêm vào FAISS với ID: {faiss_id}")
            except Exception as e:
                print(f"❌ Lỗi khi xử lý chunk {chunk_paths}: {e}")
        self.save_faiss_index()
        