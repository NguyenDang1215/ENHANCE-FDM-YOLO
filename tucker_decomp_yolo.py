import torch
import torch.nn as nn
from ultralytics import YOLO

def tucker_decompose_conv_layer(conv_layer, rank_ratio=0.5):
    """
    Phân tích Tucker 100% thuần PyTorch - Bỏ qua hoàn toàn TensorLy
    Đảm bảo không bao giờ bị lỗi Tuple hay Shape Mismatch.
    """
    in_channels = conv_layer.in_channels
    out_channels = conv_layer.out_channels
    
    # 1. Tính toán Ranks (Giữ lại số lượng kênh sau nén)
    R_o = max(1, int(out_channels * rank_ratio))
    R_i = max(1, int(in_channels * rank_ratio))
    
    # Kéo weight về CPU và ép kiểu float32 để tính SVD an toàn nhất
    weight = conv_layer.weight.data.cpu().float()
    O, I, H, W = weight.shape
    
    # ==========================================
    # TOÁN HỌC TUCKER (HOSVD) THUẦN PYTORCH
    # ==========================================
    
    # 2. SVD cho Mode 0 (Output channels)
    W_mod0 = weight.view(O, -1) # Dàn phẳng tensor theo chiều Output
    U_o, S_o, V_o = torch.linalg.svd(W_mod0, full_matrices=False)
    U_o = U_o[:, :R_o]  # Cắt lấy R_o cột quan trọng nhất. Shape: (O, R_o)
    
    # 3. SVD cho Mode 1 (Input channels)
    W_mod1 = weight.permute(1, 0, 2, 3).reshape(I, -1) # Dàn phẳng theo chiều Input
    U_i, S_i, V_i = torch.linalg.svd(W_mod1, full_matrices=False)
    U_i = U_i[:, :R_i]  # Cắt lấy R_i cột quan trọng nhất. Shape: (I, R_i)
    
    # 4. Tính Core Tensor bằng phép chiếu Einsum
    # Công thức: Chiếu tensor gốc lên 2 ma trận đặc trưng U_o và U_i
    core = torch.einsum('oihw, oO, iI -> OIhw', weight, U_o, U_i)
    
    # ==========================================
    # TÁI TẠO KIẾN TRÚC MẠNG
    # ==========================================
    
    # Lớp 1: Pointwise Conv (Giảm chiều in_channels -> R_i)
    first_layer = nn.Conv2d(in_channels=I, out_channels=R_i, kernel_size=1, bias=False)
    first_layer.weight.data = U_i.t().unsqueeze(-1).unsqueeze(-1).contiguous()
    
    # Lớp 2: Spatial Conv (Cốt lõi xử lý không gian R_i -> R_o)
    core_layer = nn.Conv2d(in_channels=R_i, out_channels=R_o, kernel_size=conv_layer.kernel_size,
                           stride=conv_layer.stride, padding=conv_layer.padding, bias=False)
    core_layer.weight.data = core.contiguous()
    
    # Lớp 3: Pointwise Conv (Phục hồi chiều R_o -> out_channels)
    last_layer = nn.Conv2d(in_channels=R_o, out_channels=O, kernel_size=1, bias=conv_layer.bias is not None)
    last_layer.weight.data = U_o.unsqueeze(-1).unsqueeze(-1).contiguous()
    if conv_layer.bias is not None:
        last_layer.bias.data = conv_layer.bias.data
        
    # Gom 3 lớp lại và trả về đúng bộ nhớ gốc của mô hình
    device = conv_layer.weight.device
    dtype = conv_layer.weight.dtype
    seq_module = nn.Sequential(first_layer, core_layer, last_layer)
    seq_module.to(device).to(dtype)
    
    return seq_module

def apply_tucker_to_model(module, rank_ratio=0.5):
    """Đệ quy nén các lớp Conv2d"""
    for name, child in module.named_children():
        # Lọc bỏ các lớp Depthwise (groups > 1) và lớp quá nhỏ (<=16 channels)
        if isinstance(child, nn.Conv2d) and child.groups == 1 and child.in_channels > 16:
            print(f"Đang nén layer: {name} | Size: {child.in_channels} -> {child.out_channels}")
            new_layer = tucker_decompose_conv_layer(child, rank_ratio)
            setattr(module, name, new_layer)
        else:
            apply_tucker_to_model(child, rank_ratio)

if __name__ == "__main__":
    # Đường dẫn file best.pt
    base_model_path = r"C:\hoc\KLTN\YOLO-APD1-main\result\result\result_fdm_p2_simam_20epoch\runs_train\fdm_kitti_v13\weights\best.pt"
    
    print(f"Loading base model: {base_model_path}")
    model = YOLO(base_model_path)
    pytorch_model = model.model
    
    print("========================================")
    print(" Bắt đầu thực hiện Tucker (Thuần PyTorch) ")
    print("========================================")
    
    # Thực hiện nén
    apply_tucker_to_model(pytorch_model, rank_ratio=0.5)
    
    compressed_model_path = "yolov8_tucker.pt"
    torch.save(model.ckpt, compressed_model_path)
    
    print("========================================")
    print(f"Hoàn tất tuyệt đối! Model nén được lưu tại: {compressed_model_path}")