# SteelDefect AI

使用原 ResNet 的网络结构和上一级目录的 `ResNet_best_acc.pth`，在本机识别六种钢材表面缺陷。

## 启动

在 PowerShell 中运行以下命令。虚拟环境可复用已有的 PyTorch 和 torchvision，并将新安装的 Gradio 依赖保留在 Demo 目录内。

```powershell
cd 'D:\idea\project\ResNet\demo'
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

后续启动只需执行最后一条命令。打开终端显示的本地网址（通常为 http://127.0.0.1:7860）。左侧初始展示 NEU-CLS 的缺陷样本拼图。点击“上传钢材图片”，选择单张 JPG、PNG、BMP 或 WebP，右侧将按置信度显示六类预测结果。

输入按训练流程转成单通道灰度图并缩放至 200×200。模型使用 CUDA（可用时）或 CPU 推理。
