# import torch
# from torchvision import transforms, models
# from PIL import Image
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework import status
# import os
#
# # Load the pre-trained model and accuracy
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_path = 'pytorchToDjangoTest/vit_model.pth'
# model_info = torch.load(model_path, map_location=device)
#
# # Define the number of classes (make sure this matches your dataset)
# num_classes = 5
# class_names = ['cat', 'dinos', 'dog', 'squirtle', 'tibetfox']  # Modify according to your classes
#
# model = models.resnet18(pretrained=False)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, num_classes)  # Ensure the output layer matches the number of classes
# model.load_state_dict(model_info['model_state_dict'])
# model = model.to(device)
# model.eval()
#
# # Assuming the test accuracy is stored in model_info
# test_acc = model_info['test_acc'].item()
#
# # Define the data transformations
# data_transforms = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#
# class ImageClassificationView(APIView):
#     parser_classes = (MultiPartParser, FormParser)
#
#     def post(self, request, *args, **kwargs):
#         if 'image' not in request.FILES:
#             return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)
#
#         image = request.FILES['image']
#         img = Image.open(image)
#         img = data_transforms(img).unsqueeze(0).to(device)
#
#         with torch.no_grad():
#             outputs = model(img)
#             _, preds = torch.max(outputs, 1)
#
#         predicted_class = preds.item()
#         class_name = class_names[predicted_class]
#
#         return Response({"predicted_class": class_name, "accuracy": test_acc})