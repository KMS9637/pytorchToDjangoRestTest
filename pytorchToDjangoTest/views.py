from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageSerializer
import torch
from torchvision import transforms
from PIL import Image
import io
from torch import nn

class SuperLightMobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(SuperLightMobileNet, self).__init__()
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        self.num_classes = num_classes
        self.model = nn.Sequential(
            conv_bn(  3,  16, 2),
            conv_dw( 16,  32, 1),
            conv_dw( 32, 64, 2),
            conv_dw(64, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 1)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)
    def forward(self, x):
        x = self.model(x)
        x = self.gap(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SuperLightMobileNet(3).to(device)
# 모델 로드 (미리 로드해 두기)
# model = torch.load('pytorchToDjangoTest/model_30.pth',map_location=torch.device('cpu'))
state_dict = torch.load('pytorchToDjangoTest/model_30_team2.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()


class ImageClassificationView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            print(f'image: {image}')

            # 이미지 변환
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # image = Image.open(image)
            image = Image.open(image).convert('RGB')
            image = transform(image).unsqueeze(0)

            class_labels = {
                0: '현무',
                1: '사암',
                2: '화강암',

            }  # Example labels

            # 모델 예측
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                predicted_class_index = predicted.item()
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_class_index].item()
                predicted_class_label = class_labels[predicted_class_index]
            response_data = {
                'predicted_class_index': predicted_class_index,
                'predicted_class_label': predicted_class_label,
                'confidence': confidence
            }

            return Response(response_data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
