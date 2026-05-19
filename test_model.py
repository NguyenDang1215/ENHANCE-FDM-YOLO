import torch
import traceback
from ultralytics import YOLO

def test_custom_yolo():
    # Tên file YAML mà bạn đã tạo ở bước trước
    yaml_path = 'yolo8_goldacsim_fdm_p2_simam.yaml'
    
    print("="*60)
    print("🚀 BẮT ĐẦU TEST KIẾN TRÚC FDM-YOLO x GOLDACSIM")
    print("="*60)

    # ---------------------------------------------------------
    # BƯỚC 1: KIỂM TRA PARSING YAML
    # ---------------------------------------------------------
    print(f"\n[1/3] Đang đọc file cấu hình: {yaml_path}...")
    try:
        model = YOLO(yaml_path)
        print("✅ THÀNH CÔNG: Ultralytics đã đọc hiểu file YAML và khởi tạo các layer (Fast-C2f, EMA)!")
        
        # Lấy thông tin tổng quan của mạng (Params, GFLOPs)
        model.info()
        
    except Exception as e:
        print("\n❌ LỖI BƯỚC 1 (Khởi tạo mạng):")
        print("Nguyên nhân thường gặp: Quên khai báo tên class trong tasks.py hoặc __init__.py.")
        traceback.print_exc()
        return

    # ---------------------------------------------------------
    # BƯỚC 2: KIỂM TRA FORWARD PASS (LUỒNG TENSOR)
    # ---------------------------------------------------------
    print("\n[2/3] Đang chạy thử luồng dữ liệu (Forward Pass) với ảnh giả 640x640...")
    try:
        # Tạo một tensor ảnh ngẫu nhiên: Batch=1, Channels=3, Height=640, Width=640
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # Tự động chọn GPU nếu có, không thì dùng CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   -> Đang test trên thiết bị: {device}")
        
        model.to(device)
        dummy_input = dummy_input.to(device)

        pytorch_model = model.model
        pytorch_model.eval() # Chuyển sang chế độ eval để tắt Dropout/BatchNorm ảo
        
        with torch.no_grad():
            outputs = pytorch_model(dummy_input)
        
        print("✅ THÀNH CÔNG: Dữ liệu chảy qua các nhánh SimFusion, Fast-C2f, EMA mượt mà!")
        
        # In thử shape đầu ra để xem Head dự đoán bbox có đúng kích thước không
        if isinstance(outputs, (list, tuple)):
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    print(f"   -> Nhánh output {i} shape: {out.shape}")
        else:
            print(f"   -> Output shape: {outputs.shape}")

    except Exception as e:
        print("\n❌ LỖI BƯỚC 2 (Lệch Tensor / Lỗi tính toán kênh):")
        print("Nguyên nhân thường gặp: Lớp Concat hoặc Sum nhận 2 tensor khác kích thước (C, H, W).")
        traceback.print_exc()
        return

    # ---------------------------------------------------------
    # BƯỚC 3: KIỂM TRA PIPELINE HUẤN LUYỆN
    # ---------------------------------------------------------
    print("\n[3/3] Đang test quá trình tính Loss & Backpropagation với dataset coco8...")
    try:
        # Chạy 1 epoch với dataset siêu nhẹ có sẵn của Ultralytics
        results = model.train(
            data='coco8.yaml', 
            epochs=1, 
            imgsz=640, 
            batch=2, 
            device=device,
            project='runs_test', # Đẩy kết quả test ra một thư mục riêng biệt
            name='test_fdm',
            exist_ok=True,
            workers=0 # Tắt đa luồng để test nhanh và tránh lỗi bộ nhớ trên Windows
        )
        print("✅ THÀNH CÔNG: Mô hình có thể huấn luyện, cập nhật trọng số (Gradient) bình thường!")
    except Exception as e:
        print("\n❌ LỖI BƯỚC 3 (Quá trình Train/Loss):")
        traceback.print_exc()
        return

    print("\n" + "="*60)
    print("🎉 TẤT CẢ CÁC BƯỚC TEST ĐÃ PASS! MÔ HÌNH CỦA BẠN SẴN SÀNG ĐỂ TRAIN CHÍNH THỨC!")
    print("="*60)

if __name__ == '__main__':
    test_custom_yolo()