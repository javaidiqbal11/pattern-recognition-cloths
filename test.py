import torch
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder

# Define the transformations for the image
data_dir = 'romo'

transform = transforms.Compose([
    transforms.Resize((150, 150)),  # Resize to 224x224 (adjust as needed)
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = ImageFolder(data_dir+'/train', transform=transform)
# Detect the available device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model (assuming it's saved as 'model.pth')
input_size = (150, 150)
num_classes = 18
model = torch.load('/kaggle/working/pr.pth')
model.to(device)  # Move the model to the detected device
model.eval()  # Set the model to evaluation mode


# Function to predict the class of an unknown image
def predict_image(image_path, model, transform, class_names, device):
    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)  # Move the image to the detected device

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Get the predicted class name
    predicted_class = class_names[predicted.item()]
    return predicted_class


# Example usage
image_path = 'image.jpg'  # Replace with the path to your image
class_names = dataset.classes  # Replace with your class names if different

predicted_class = predict_image(image_path, model, transform, class_names, device)
print(f'The predicted class for the image is: {predicted_class}')
