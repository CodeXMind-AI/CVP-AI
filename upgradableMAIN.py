import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Advanced Data Augmentation for Training (التحويلات العشوائية القوية لمحاكاة الخطوط البشرية)
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),                                   # تدوير واسع النطاق
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.7, 1.3)), # إزاحة أكبر وتغيير حجم مرن
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),                 # تشوه هندسي لمحاكاة زوايا الكاميرا
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Test data stays clean (بيانات الاختبار تبقى نقية للمقارنة الحقيقية)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 2. Load MNIST Dataset with Augmentation
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 3. Define the CNN Architecture (بنية الشبكة العصبية)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Device configuration (CPU)
device = torch.device('cpu')
print(f"🚀 Using device: {device}")

model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop (عملية التدريب لـ 15 Epochs بضبط تام)
num_epochs = 15  
print("=== Starting Robust Training with Aggressive Data Augmentation ===")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 5. Evaluate Accuracy on Test Set
model.eval()
correct = 0  
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"🎯 Test Accuracy after Robust Augmentation: {100 * correct / total:.2f}%")

# 6. Save the newly trained weights (حفظ الـ Weights الجداد بامتياز)
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
print("=== Model weights saved successfully as 'mnist_cnn_model.pth'! ===")