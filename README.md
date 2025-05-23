# Mori - Hệ Thống Tìm Kiếm Thông Minh

Mori là một công cụ tìm kiếm tiên tiến được xây dựng bằng Django và các công nghệ AI, được thiết kế để cung cấp khả năng tìm kiếm thông minh sử dụng ngữ nghĩa hình ảnh và văn bản.

## Tổng Quan Dự Án

Mori kết hợp nhiều công nghệ tiên tiến:
- Django REST framework cho các dịch vụ backend
- PostgreSQL để lưu trữ dữ liệu
- FAISS cho tìm kiếm tương đồng hiệu quả
- OpenCLIP và Sentence Transformers cho tìm kiếm dựa trên AI
- Docker và Docker Compose để containerization

## Yêu Cầu Hệ Thống

- Docker và Docker Compose
- Git

## Hướng Dẫn Cài Đặt

### Bắt Đầu Nhanh

1. Clone repository:
   ```
   git clone <repository-url>
   cd mori
   ```

2. Tạo file `.env` trong thư mục gốc của dự án với các biến sau:
   ```
   POSTGRES_USER=your_postgres_user
   POSTGRES_PASSWORD=your_postgres_password
   POSTGRES_DB=mori_db
   POSTGRES_HOST=mori_db
   POSTGRES_PORT=5432
   ```

3. Xây dựng và khởi động các container:
   ```
   docker compose down -v        # Dọn dẹp các container hiện có
   docker volume prune -f        # Xóa các volume không sử dụng
   docker compose up -d --build  # Xây dựng và khởi động ở chế độ ngầm
   ```

4. Truy cập ứng dụng:
   - Giao diện web: http://localhost:8000/
   - Tài liệu API: http://localhost:8000/swagger/

### Chi Tiết Cài Đặt

Dự án bao gồm ba dịch vụ chính:
- `mori_search`: Ứng dụng Django chính chạy trên cổng 8000
- `mori_db`: Cơ sở dữ liệu PostgreSQL chạy trên cổng 5435
- `trash_cron`: Một dịch vụ thực hiện các tác vụ dọn dẹp định kỳ

## Phát Triển

Để thực hiện thay đổi trong dự án:
1. Dừng các container: `docker compose down`
2. Thực hiện các thay đổi của bạn
3. Xây dựng lại và khởi động lại: `docker compose up -d --build`

## Xử Lý Sự Cố

- Nếu ứng dụng không khởi động được, kiểm tra logs: `docker compose logs mori_search`
- Vấn đề kết nối cơ sở dữ liệu có thể cần thiết lập lại volumes: `docker compose down -v && docker volume prune -f`
- Đối với vấn đề quyền truy cập với script: `chmod +x mori.sh`

## Những Điều Cần Lưu Ý Khi Triển Khai Cloud

- Đảm bảo các cổng 8000 và 5435 được mở trong nhóm bảo mật/tường lửa cloud của bạn
- Đối với môi trường sản xuất, sửa đổi cài đặt Django để sử dụng biến môi trường cho thông tin nhạy cảm
- Cân nhắc sử dụng dịch vụ cơ sở dữ liệu được quản lý thay vì PostgreSQL trong container
- Thiết lập chiến lược sao lưu phù hợp cho cơ sở dữ liệu của bạn
- Cấu hình SSL/TLS phù hợp để đảm bảo liên lạc an toàn