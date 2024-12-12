import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time



from pynvml import *
import sys

def cal_GPU():
    nvmlInit()
        # 使用するGPUのインデックス（0を指定すると最初のGPU）
    gpu_index = 0
    try:
        # GPUのハンドルを取得
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        print(f"GPU {gpu_index}: {nvmlDeviceGetName(handle).decode('utf-8')}")

        # 実行中のプロセス情報を取得
        processes = nvmlDeviceGetComputeRunningProcesses(handle)
        if len(processes) == 0:
            print("No processes found running on the GPU.")
        else:
            # 各プロセスの情報を表示
            print(f"{'PID':>10}  {'Process_name':>20} {'Memory Usage (MB)':>20} {'Power Usage (W)':>20} {'Temperature (℃)' :>20} {'GPU Utilization (%)'} {'Memory Utilization (%)'}")
            for process in processes:
                pid = process.pid
                memory_usage = process.usedGpuMemory / (1024 ** 2)  # メモリ使用量をMB単位に変換
                  # プロセス名の取得（pidを使ってシステムから取得）
                try:
                    with open(f"/proc/{pid}/cmdline", "r") as f:
                        process_name = f.read().strip().replace('\x00', ' ')
                except FileNotFoundError:
                    process_name = "Unknown"
                
                power_usage = nvmlDeviceGetPowerUsage(handle) / 1000  # ミリワットをワットに変換
                temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                utilization = nvmlDeviceGetUtilizationRates(handle)

                
                
                print(f"{pid:>10}   {process_name:>20} {memory_usage:>20.2f} MB  {power_usage:>20} {temperature:>20.2f}  {utilization.gpu:>20} {utilization.memory:>20.2f}")

    except NVMLError as error:
        print(f"NVML error: {error}")
        sys.exit(1)

    # NVMLの終了処理
    nvmlShutdown()



    
    




# GPUが利用可能か確認し、デバイスを設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# パラメータ設定
imageWH = 28  # MNISTの画像サイズは28x28
channel = 1  # MNISTはグレースケール画像

# ViTハイパーパラメータ
patchWH = 7  # 28x28の画像を7x7のパッチに分割
splitRow = imageWH // patchWH  # 4
splitCol = imageWH // patchWH  # 4
patchTotal = (imageWH // patchWH) ** 2  # 4 * 4 = 16
patchVectorLen = channel * (patchWH ** 2)  # 1 * 49 = 49


embedVectorLen = 128  # 埋め込み後のベクトルの次元
head = 8
dim_feedforward = 128
layers = 6
batch_size=128

# Transformerレイヤーハイパーパラメータ


activation = "gelu"


# Vision Transformerモデルの定義
class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.patchEmbedding = nn.Linear(patchVectorLen, embedVectorLen)
        self.cls = nn.Parameter(torch.zeros(1, 1, embedVectorLen))
        self.positionEmbedding = nn.Parameter(torch.zeros(1, patchTotal + 1, embedVectorLen))
        encoderLayer = TransformerEncoderLayer(
            d_model=embedVectorLen,
            nhead=head,
            dim_feedforward=dim_feedforward,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        self.transformerEncoder = TransformerEncoder(encoderLayer, layers)
        self.mlpHead = nn.Linear(embedVectorLen, 10)  # MNISTは10クラス分類

    def patchify(self, img):
        # 画像を7x7のパッチに分割
        horizontal = torch.stack(torch.chunk(img, splitRow, dim=2), dim=1)
        patches = torch.cat(torch.chunk(horizontal, splitCol, dim=4), dim=1)
        return patches

    def forward(self, x):
        x = self.patchify(x)
        x = torch.flatten(x, start_dim=2)
        x = self.patchEmbedding(x)
        clsToken = self.cls.repeat_interleave(x.shape[0], dim=0)
        x = torch.cat((clsToken, x), dim=1)
        x += self.positionEmbedding
        x = self.transformerEncoder(x)
        x = self.mlpHead(x[:, 0, :])
        return x

# データセットの前処理と読み込み
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# モデルのインスタンス化とGPUへの転送
model = ViT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# トレーニング関数
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # 画像とラベルをGPUに転送
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
      


# テスト関数
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # 画像とラベルをGPUに転送
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total}%')
    
   

# トレーニングとテストの実行
start = time.time()

train(model, train_loader, criterion, optimizer, epochs=1)
test(model, test_loader)

end = time.time()
print(f"実行時間：{end - start}秒")

print('file1で動かしたときの最終的なGPU使用量')
cal_GPU()

