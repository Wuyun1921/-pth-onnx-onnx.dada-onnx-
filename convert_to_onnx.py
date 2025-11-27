import torch
import torch.nn as nn
from torchvision import models
import os

# ==========================================================
# 1. 模型定义 (必须与训练时的定义完全一致)
# ==========================================================
class RegressionResNet50(nn.Module):
    def __init__(self, pretrained=False): # 转换时不需要预训练权重，只需要结构
        super().__init__()
        # 注意：这里 weights 设置为 None，因为我们会加载你训练好的 .pth 权重
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)

def convert_to_onnx():
    # ==========================================================
    # 2. 配置路径和参数
    # ==========================================================
    model_path = "best_resnet50_regression.pth"
    onnx_path = "best_resnet50_regression.onnx"
    
    # 输入尺寸: (Batch_Size, Channels, Height, Width)
    # 你的训练脚本中使用的是 224x224
    input_shape = (1, 3, 224, 224) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Check file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    # ==========================================================
    # 3. 加载模型
    # ==========================================================
    print("Loading model structure...")
    model = RegressionResNet50(pretrained=False)
    
    print(f"Loading weights from {model_path}...")
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.to(device)
    model.eval() # 必须设置为评估模式，否则 BatchNormalization 等层行为不正确

    # ==========================================================
    # 4. 导出 ONNX
    # ==========================================================
    print("Creating dummy input...")
    dummy_input = torch.randn(input_shape).to(device)

    print(f"Exporting to {onnx_path}...")
    try:
        # 1. 先导出模型
        torch.onnx.export(
            model,                      # 模型实例
            dummy_input,                # 虚拟输入
            onnx_path,                  # 输出文件路径
            verbose=False,              # 是否打印详细信息
            input_names=['input'],      # 输入节点名称
            output_names=['output'],    # 输出节点名称
            opset_version=11,           # ONNX opset 版本
            dynamic_axes={              # 动态轴
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✅ Initial export successful to: {onnx_path}")

        # 2. 尝试合并 .data 文件 (如果有生成)
        # 某些 PyTorch 版本或配置会生成 .onnx 和 .onnx.data 两个文件
        # 这里我们使用 onnx 库重新保存一次，将其合并为一个文件
        try:
            import onnx
            print("🔄 Checking for split files and merging if necessary...")
            
            # 加载导出的模型（会自动加载关联的 .data 文件）
            onnx_model = onnx.load(onnx_path)
            
            # 重新保存（onnx.save 默认会将权重嵌入到模型文件中，除非模型 > 2GB）
            onnx.save(onnx_model, onnx_path)
            
            # 检查并删除可能存在的 .data 文件
            data_file = onnx_path + ".data"
            if os.path.exists(data_file):
                os.remove(data_file)
                print(f"🗑️ Removed external data file: {data_file}")
            
            print(f"✅ Merged into single file: {onnx_path}")
            
            # 验证模型
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model check passed.")
            
        except ImportError:
            print("⚠️ 'onnx' library not found. If you see a .data file, install 'onnx' (pip install onnx) and run this again to merge them.")
        except Exception as e:
            print(f"⚠️ Merge/Check process failed (model is still usable): {e}")

    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    convert_to_onnx()