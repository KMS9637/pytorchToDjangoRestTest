from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageSerializer
import torch
from torchvision import transforms, models
from PIL import Image
import io
from torch import nn

# 경로 설정
model_weight_save_path = "pytorchToDjangoTest/resnet50_epoch_50_loss_1101_acc_9266.pth"
num_classes = 10

# ResNet-50 모델 정의 및 로드
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)

# 모델 가중치 로드
checkpoint = torch.load(model_weight_save_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint, strict=False)
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

class ImageClassificationView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']

            # 이미지 변환
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # 이미지 처리
            image = Image.open(image).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            # 예측
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                predicted_class_index = predicted.item()
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_class_index].item()
                # 1조
                # class_labels = {0: '고양이', 1: '공룡', 2: '강아지',3: '꼬북이',4: '티벳여우'}
                #3조
                class_labels = {0: '공구톱', 1: '공업용가위', 2: '그라인더', 3: '니퍼', 4: '드라이버'
                                , 5: '망치', 6: '스패너', 7: '전동드릴', 8: '줄자', 9: '버니어 캘리퍼스'}
                #2조
                # class_labels = {0: '업소용냉장고', 1: 'cpu', 2: '드럼세탁기', 3: '냉장고', 4: '그래픽카드', 5: '메인보드'
                #     , 6: '전자레인지', 7: '파워', 8: '렘', 9: '스탠드에어컨', 10: 'TV', 11: '벽걸이에어컨', 12: '통돌이세탁기'}

                predicted_class_label = class_labels[predicted_class_index]

            # 응답 데이터
            response_data = {
                'predicted_class_index': predicted_class_index,
                'predicted_class_label': predicted_class_label,
                'confidence': confidence
            }

            return Response(response_data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)