import torch
from torchvision import transforms, models
from PIL import Image
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
import os
import timm

from pytorchToDjangoTest.serializers import ImageSerializer

# Load the pre-trained model and accuracy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=5)
vit_model.load_state_dict(torch.load('pytorchToDjangoTest/vit_model.pth'))
vit_model = vit_model.to(device)

deit_model = timm.create_model('deit_base_patch16_224', pretrained=True, num_classes=5)
deit_model.load_state_dict(torch.load('pytorchToDjangoTest/deit_model.pth.pth'))
deit_model = deit_model.to(device)

vit_model.eval()
deit_model.eval()



# Define the number of classes (make sure this matches your dataset)
num_classes = 5
class_names = ['cat', 'dinos', 'dog', 'squirtle', 'tibetfox']  # Modify according to your classes

# Assuming the test accuracy is stored in model_info
# test_acc = model_info['test_acc'].item()

# Define the data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        vit_output = vit_model(image)
        deit_output = deit_model(image)

    vit_pred_idx = torch.argmax(vit_output, dim=1).item()
    deit_pred_idx = torch.argmax(deit_output, dim=1).item()

    vit_pred_class = class_names[vit_pred_idx]
    deit_pred_class = class_names[deit_pred_idx]

    # Dummy true labels for illustration; replace with actual labels in practice
    true_label_idx = 0  # Example true label
    true_label_class = class_names[true_label_idx]

    vit_correct = vit_pred_idx == true_label_idx
    deit_correct = deit_pred_idx == true_label_idx

    vit_accuracy = int(vit_correct)
    deit_accuracy = int(deit_correct)

    return {
        'vit_prediction': vit_pred_class,
        'deit_prediction': deit_pred_class,
        'vit_accuracy': vit_accuracy,
        'deit_accuracy': deit_accuracy

            }

class ImageClassificationView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            predictions = predict(image)
            return Response(predictions, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)