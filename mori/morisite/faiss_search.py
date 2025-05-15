import os
import faiss
import torch
import numpy as np
from PIL import Image
import open_clip
import time
from django.conf import settings
from .models import Photo
from .serializers import *
from .utils import translate_text
from sentence_transformers import SentenceTransformer

class FaissSearch:
    def __init__(self, user=None, model_name='clip-ViT-B-32', feature_dim=512):
        self.user = user
        self.device = "cpu"
        self.feature_dim = feature_dim

        if user:
            self.index_path = os.path.join(settings.MEDIA_ROOT, f"faiss_index_user_{user.id_user}.bin")
        else:
            self.index_path = os.path.join(settings.MEDIA_ROOT, "faiss_index_global.bin")

        self.model = SentenceTransformer(model_name, device=self.device)
        if os.path.exists(self.index_path):
            print(f"✅ FAISS index loaded từ {self.index_path}")
            self.index = faiss.read_index(self.index_path)
        else:
            print(f"⚠️ Không tìm thấy FAISS index tại {self.index_path}, tạo FAISS index mới.")
            self.index = faiss.IndexFlatIP(self.feature_dim)
            self.save_faiss_index()

    def save_faiss_index(self):
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            print(f"✅ FAISS index đã được lưu vào {self.index_path}")
        except Exception as e:
            print(f"❌ Lỗi khi lưu FAISS index: {e}")

    def _extract_features(self, query, mode="text"):
        """Trích xuất đặc trưng cho văn bản hoặc hình ảnh."""
        try:
            if mode == "text":
                query = str(translate_text(query, to_lang='en'))
                print("query: ", query)
                features = self.model.encode([query], convert_to_numpy=True)
            elif mode == "image":
                image = Image.open(query).convert("RGB")
                features = self.model.encode([image], convert_to_numpy=True)
            else:
                raise ValueError("❌ Mode không hợp lệ!")
            features = features / np.linalg.norm(features, axis=-1, keepdims=True)
            return features.astype(np.float32)
        except Exception as e:
            print(f"❌ Lỗi khi trích xuất đặc trưng: {e}")
            return None

    def _get_photos_from_indices(self, indices):
        print("indices: ", indices)
        photo_ids = [int(i) for i in indices if i >= 0]
        print("photo_ids: ", photo_ids)
        photos = Photo.objects.filter(
            faiss_id_public__in=photo_ids,
            is_public=True,
            is_deleted=False
        )
        return PhotoCommunitySerializer(photos, many=True).data

    def _get_photos_from_indices_for_user(self, indices):
        try:
            print("indices: ", indices)
            photo_ids = [int(i) for i in indices if i >= 0]
            print("photo_ids: ", photo_ids)
            photos = Photo.objects.filter(
                faiss_id__in=photo_ids,
                album__user=self.user,
                is_deleted=False
            )
            return PhotoInviCommunitySerializer(photos, many=True).data
        except Exception as e:
            print(f"❌ Lỗi khi lấy ảnh từ chỉ số: {e}")
            return []

    # Tìm kiếm theo user
    def search_for_user(self, query, mode="text", k=5):
        start_time = time.perf_counter()
        print(f"search cho user {self.user}")
        if not self.user:
            raise ValueError("❌ User không hợp lệ!")
        print("mode: ", mode)
        query_features = self._extract_features(query, mode)
        if query_features is None:
            return []
        
        scores, idx_images = self.index.search(query_features, k=k)
        print("scores: ", scores.flatten())
        print("id images: ", idx_images.flatten())
        results = self._get_photos_from_indices_for_user(idx_images.flatten().tolist()[::-1])
        end_time = time.perf_counter()
        print(f"✅ Thời gian model truy xuất bin: {end_time - start_time:.5f} giây")
        return list(results)

    # Tìm kiếm toàn cục
    def search_global(self, query, mode="text", k=5):
        start_time = time.perf_counter()
        print("search toàn cục")
        print("query: ", query)
        print("mode: ", mode)
        query_features = self._extract_features(query, mode)
        if query_features is None:
            return []   
        scores, idx_images = self.index.search(query_features, k=k)
        print("scores: ", scores.flatten())
        print("id images: ", idx_images.flatten())
        print("id images reserve: ",idx_images.flatten().tolist()[::-1])
        results = self._get_photos_from_indices(idx_images.flatten().tolist()[::-1])
        print("results: ", results)
        end_time = time.perf_counter()
        print(f"✅ Thời gian truy xuất bin: {end_time - start_time:.5f} giây")
        return results
