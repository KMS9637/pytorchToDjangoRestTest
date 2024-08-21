from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageSerializer
import torch
from torchvision import transforms, models
from PIL import Image
import io
from torch import nn
import numpy as np
import torch.nn.functional as F

# 경로 설정
model_weight_save_path = "pytorchToDjangoTest/resnet50_epoch_10_240821_acc91.pth"
num_classes = 2

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


# 이미지에서 임베딩 벡터 추출
def extract_features(image, model, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        features = model(image)
    return features


# 유클리드 거리 계산 함수
def calculate_distance(embedding1, embedding2):
    return torch.dist(embedding1, embedding2)


# 임베딩 벡터를 이용한 거리 기반 분류 함수
def classify_based_on_distance(image_embedding, class_embeddings, threshold=1.0):
    min_distance = float('inf')
    best_class = None

    for class_label, class_embedding in class_embeddings.items():
        distance = calculate_distance(image_embedding, class_embedding)
        if distance < min_distance:
            min_distance = distance
            best_class = class_label

    if min_distance > threshold:
        return "기타"
    else:
        return best_class

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

            # # 임베딩 벡터 추출
            # image_embedding = extract_features(image, model, device)
            #
            # # 학습된 클래스별 평균 임베딩 벡터 (사전 준비 필요)
            # class_embeddings = {
            #     "고양이": torch.tensor([0.1, 0.2, 0.3]),  # 실제 임베딩 벡터로 대체
            #     "공룡": torch.tensor([0.4, 0.5, 0.6]),  # 실제 임베딩 벡터로 대체
            #     "강아지": torch.tensor([0.7, 0.8, 0.9]),  # 실제 임베딩 벡터로 대체
            #     "꼬북이": torch.tensor([0.2, 0.3, 0.4]),  # 실제 임베딩 벡터로 대체
            #     "티벳여우": torch.tensor([0.5, 0.6, 0.7])  # 실제 임베딩 벡터로 대체
            # }
            #
            # # 임베딩을 이용한 거리 기반 분류
            # predicted_class_label = classify_based_on_distance(image_embedding, class_embeddings, threshold=1.0)
            #
            # # 모든 클래스에 대한 거리 반환
            # class_distances = {label: round(calculate_distance(image_embedding, embedding).item(), 4) for
            #                    label, embedding in class_embeddings.items()}

            # 예측
            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                _, predicted = torch.max(outputs, 1)
                predicted_class_index = predicted.item()
                # confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_class_index].item()
                confidence = probabilities[predicted_class_index].item()
                max_confidence, predicted = torch.max(probabilities, 0)


                # 1조
                # class_labels = {0: '고양이', 1: '공룡', 2: '강아지',3: '꼬북이',4: '티벳여우'}
                #3조
                class_labels = {0: '망치', 1: '공업용가위'}
                # class_labels = {0: '공구톱', 1: '공업용가위', 2: '그라인더', 3: '니퍼', 4: '드라이버'
                #                 , 5: '망치', 6: '스패너', 7: '전동드릴', 8: '줄자', 9: '버니어 캘리퍼스'}
                #2조
                # class_labels = {0: '업소용냉장고', 1: 'cpu', 2: '드럼세탁기', 3: '냉장고', 4: '그래픽카드', 5: '메인보드'
                #     , 6: '전자레인지', 7: '파워', 8: '렘', 9: '스탠드에어컨', 10: 'TV', 11: '벽걸이에어컨', 12: '통돌이세탁기'}

                # predicted_class_label = class_labels[predicted_class_index]
                # 정확도가 50% 미만인 경우 기타로 분류

                # OpenMax를 사용하여 예측
                # predicted_class, confidence = predict_with_openmax(image, model, weibull_model, class_labels)

                if max_confidence < 0.5:
                    predicted_class_label = "기타"
                else:
                    # predicted_class_label = class_labels[predicted_class_index]
                    predicted_class_label = class_labels.get(predicted.item(), "기타")

                # 모든 클래스에 대한 확률 반환
                class_confidences = {class_labels[i]: round(probabilities[i].item(), 4) for i in range(num_classes)}

            # 응답 데이터
            response_data = {
                # 'class_distances': class_distances,  # 각 클래스에 대한 거리
                'predicted_class_index': predicted_class_index,
                'predicted_class_label': predicted_class_label,
                'confidence': confidence,
                'class_confidences': class_confidences  # 각 클래스에 대한 확률
            }

            return Response(response_data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)