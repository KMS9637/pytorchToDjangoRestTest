# import torch
# from torchvision import transforms, models
# from PIL import Image
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework import status
# import os
# import timm
# import torch.nn.functional as F
#
# from pytorchToDjangoTest.serializers import ImageSerializer
#
# # Define device and class names
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_classes = 5
# class_names = ['cat', 'dinos', 'dog', 'squirtle', 'tibetfox']
#
# # Load models with the correct number of classes
# vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
# deit_model = timm.create_model('deit_base_patch16_224', pretrained=True)
#
# # Modify the head of the models to match the number of classes
# vit_model.head = torch.nn.Linear(vit_model.head.in_features, num_classes)
# deit_model.head = torch.nn.Linear(deit_model.head.in_features, num_classes)
#
# # Load the model weights with map_location to CPU, skipping the incompatible layers
# vit_checkpoint = torch.load('pytorchToDjangoTest/vit_model.pth', map_location=torch.device('cpu'))
# deit_checkpoint = torch.load('pytorchToDjangoTest/deit_model.pth', map_location=torch.device('cpu'))
#
# # Remove the incompatible layers from the checkpoints
# vit_checkpoint = {k: v for k, v in vit_checkpoint.items() if 'head' not in k}
# deit_checkpoint = {k: v for k, v in deit_checkpoint.items() if 'head' not in k}
#
# # Load the weights
# vit_model.load_state_dict(vit_checkpoint, strict=False)
# deit_model.load_state_dict(deit_checkpoint, strict=False)
#
# # Move models to the correct device
# vit_model = vit_model.to(device)
# deit_model = deit_model.to(device)
#
# vit_model.eval()
# deit_model.eval()
#
# # Define transformation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
# def predict(image_path):
#     image = Image.open(image_path)
#     image = transform(image).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         vit_output = vit_model(image)
#         deit_output = deit_model(image)
#
#     vit_pred_idx = torch.argmax(vit_output, dim=1).item()
#     deit_pred_idx = torch.argmax(deit_output, dim=1).item()
#
#     vit_pred_class = class_names[vit_pred_idx]
#     deit_pred_class = class_names[deit_pred_idx]
#
#     # Dummy true labels for illustration; replace with actual labels in practice
#     true_label_idx = 0  # Example true label
#     true_label_class = class_names[true_label_idx]
#
#     vit_correct = vit_pred_idx == true_label_idx
#     deit_correct = deit_pred_idx == true_label_idx
#
#     vit_accuracy = 100.0 if vit_correct else 0.0
#     deit_accuracy = 100.0 if deit_correct else 0.0
#
#     # Convert logits to probabilities using softmax
#     vit_probs = F.softmax(vit_output, dim=1)
#     deit_probs = F.softmax(deit_output, dim=1)
#
#     vit_pred_idx = torch.argmax(vit_probs, dim=1).item()
#     deit_pred_idx = torch.argmax(deit_probs, dim=1).item()
#
#     vit_pred_class = class_names[vit_pred_idx]
#     deit_pred_class = class_names[deit_pred_idx]
#
#     vit_match_percentage = vit_probs[0][true_label_idx].item() * 100
#     deit_match_percentage = deit_probs[0][true_label_idx].item() * 100
#
#     return {
#         'vit_prediction': vit_pred_class,
#         'deit_prediction': deit_pred_class,
#         'vit_match_percentage': f"{vit_match_percentage:.2f}%",
#         'deit_match_percentage': f"{deit_match_percentage:.2f}%",
#
#             }
#
# class ImageClassificationView(APIView):
#     def post(self, request, *args, **kwargs):
#         serializer = ImageSerializer(data=request.data)
#         if serializer.is_valid():
#             image = serializer.validated_data['image']
#             predictions = predict(image)
#             return Response(predictions, status=status.HTTP_200_OK)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)