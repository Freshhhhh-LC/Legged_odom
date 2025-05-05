import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rerun as rr
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights


# Device configuration
device = torch.device("cpu")
data_name = "2025-04-01-22-47-36"
model_name = "model_40.pth"


def gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y, max_distance):
    return (
        (np.sqrt((x - mu_x) ** 2 + (y - mu_y) ** 2) < max_distance)
        * (1 / (2 * np.pi * sigma_x * sigma_y))
        * np.exp(-((x - mu_x) ** 2 / (2 * sigma_x**2) + (y - mu_y) ** 2 / (2 * sigma_y**2)))
    )


# Custom Dataset to Load .npy Files
class CustomDataset(Dataset):
    def __init__(self, data_pattern, label_pattern):
        self.data_files = sorted(glob.glob(data_pattern))  # Load all data files
        self.label_files = sorted(glob.glob(label_pattern))  # Load all label files

        # Load all data into memory first
        self.data = [np.load(file) for file in self.data_files]  # List of numpy arrays
        self.labels = [np.load(file) for file in self.label_files]  # List of numpy arrays

        # Convert all data to PyTorch tensors and move to GPU
        self.data = [torch.tensor(d, dtype=torch.float32).permute(0, 3, 1, 2) for d in self.data]  # Shape: (N, 7, 200, 150)
        self.labels = [torch.tensor(l, dtype=torch.float32) for l in self.labels]  # Shape: (N, 3)

        # Compute actual dataset length
        self.lengths = [d.shape[0] for d in self.data]
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length  # Use the actual dataset length

    def __getitem__(self, index):
        file_idx = 0
        while index >= self.lengths[file_idx]:  # Find the correct file
            index -= self.lengths[file_idx]
            file_idx += 1

        # Get data and label from pre-loaded GPU tensors
        data = self.data[file_idx][index]  # Shape: (7, 200, 150)
        label = self.labels[file_idx][index]  # Shape: (3,)

        # Convert label to [x, y, cos(yaw), sin(yaw)]
        x, y, yaw = label
        label = torch.tensor([x, y, torch.cos(yaw), torch.sin(yaw)], dtype=torch.float32)  # Shape: (4,)

        return data, label


rr.init("train", spawn=True)
map_lines = []
map_lines.append([[-7, -4.5], [-7, 4.5], [7, 4.5], [7, -4.5], [-7, -4.5]])
map_lines.append([[0, -4.5], [0, 4.5]])
map_lines.append([[1.5 * np.cos(theta), 1.5 * np.sin(theta)] for theta in np.arange(0, 2 * np.pi + 0.1, 0.1)])
map_lines.append([[4.7, 0.0], [5.1, 0.0]])
map_lines.append([[4.9, -0.2], [4.9, 0.2]])
map_lines.append([[-5.1, 0.0], [-4.7, 0.0]])
map_lines.append([[-4.9, -0.2], [-4.9, 0.2]])
map_lines.append([[7, 1.3], [8, 1.3], [8, -1.3], [7, -1.3]])
map_lines.append([[-7, 1.3], [-8, 1.3], [-8, -1.3], [-7, -1.3]])
map_lines.append([[7, 3], [4, 3], [4, -3], [7, -3]])
map_lines.append([[-7, 3], [-4, 3], [-4, -3], [-7, -3]])
map_lines.append([[7, 2], [6, 2], [6, -2], [7, -2]])
map_lines.append([[-7, 2], [-6, 2], [-6, -2], [-7, -2]])
for line in map_lines:
    for point in line:
        point[0] = point[0] * 10 + 75
        point[1] = point[1] * 10 + 50
rr.log("field/Border", rr.LineStrips2D(map_lines, colors=[255, 255, 255]), static=True)
rr.log(
    "image/Border",
    rr.LineStrips2D(
        [((0, 0), (120, 0), (120, 150), (0, 150), (0, 0))],
        colors=[255, 255, 255],
    ),
    static=True,
)

# Load dataset
dataset = CustomDataset(f"data/{data_name}/data_batch_*.npy", f"data/{data_name}/label_batch_*.npy")
test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

# Load Pretrained ResNet-18 model and modify the first and last layers
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 150 * 100)
model = model.to(device)

# Load the model checkpoint
model.load_state_dict(torch.load(f"model/{model_name}", weights_only=True, map_location=device))
print(f"Model loaded from model/{model_name}")

# Test the model
model.eval()  # Set model to evaluation mode
x_odom = 0.0
y_odom = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        detections = {"Goalpost": [], "PenaltyPoint": [], "TCross": [], "LCross": [], "XCross": [], "Mask": []}
        key_mapping = {0: "Goalpost", 1: "PenaltyPoint", 2: "TCross", 3: "LCross", 4: "XCross", 5: "Mask", 6: "Mask"}
        non_zero_indices = torch.nonzero(images)
        for row in non_zero_indices:
            detections[key_mapping[row[1].item()]].append(row[2:].tolist())
        for i, (label, objs) in enumerate(detections.items()):
            if i == 5:
                rr.log(
                    f"image/{label}",
                    rr.Points2D([(obj[0], obj[1]) for obj in objs], labels=[f"{label}" for obj in objs], colors=[200, 200, 200]),
                )
            else:
                rr.log(
                    f"image/{label}",
                    rr.Points2D([(obj[0], obj[1]) for obj in objs], labels=[f"{label}" for obj in objs], class_ids=i),
                )

        rr.log("field/Real", rr.Points2D((labels[0, 0] * 10 + 75, labels[0, 1] * 10 + 50), radii=1.5, colors=[255, 255, 0]))
        images, labels = images.to(device), labels.to(device)
        outputs = F.softmax(model(images)).view(-1, 150, 100)
        # outputs = torch.where(outputs < 0.005, torch.tensor(1e-8), outputs)
        rr.log("field/Map", rr.DepthImage(outputs.view(150, 100).T.cpu()))
        x_grid, y_grid = torch.meshgrid(torch.arange(150), torch.arange(100))
        normal_distribution = gaussian_2d(x_grid, y_grid, x_odom * 10 + 75, y_odom * 10 + 50, 5, 5, 10)
        product_matrix = outputs.view(150, 100).cpu() + normal_distribution
        max_index = np.unravel_index(np.argmax(product_matrix), product_matrix.shape)
        x_odom = 0.1 * max_index[0] - 7.5
        y_odom = 0.1 * max_index[1] - 5
        rr.log("field/Pred", rr.Points2D((max_index[0], max_index[1]), radii=1.5, colors=[255, 0, 255]))
