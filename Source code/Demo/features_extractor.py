from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.applications.xception import Xception, preprocess_input as xception_preprocess
from tensorflow.keras.models import Model
import numpy as np

class FeatureExtractor:
    
    # Constructor (hàm khởi tạo)
    def __init__(self, arch='VGG16'):
        
        self.arch = arch
        
        # Sử dụng VGG-16 với trọng số từ ImageNet
        if self.arch == 'VGG16':
            base_model = VGG16(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        
        # Sử dụng ResNet 50 với trọng số từ ImageNet
        elif self.arch == 'ResNet50':
            base_model = ResNet50(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        
        # Sử dụng Xception với trọng số từ ImageNet
        elif self.arch == 'Xception':
            base_model = Xception(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
            
    # Phương thức trích xuất đặc trưng hình ảnh
    def extract_features(self, img):
        
        # Mô hình VGG 16 & ResNet 50 yêu cầu kích thước ảnh đầu vào là 224x224, còn Xception là 299x299
        if self.arch == 'VGG16' or self.arch == 'ResNet50':
            img = img.resize((224, 224))
        elif self.arch == 'Xception':
            img = img.resize((299, 299))
        
        # Chuyển đổi kênh ảnh sang RGB
        img = img.convert('RGB')
        
        # Chuyển đổi ảnh thành mảng (array)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        # Tiền xử lý đầu vào theo mô hình tương ứng
        if self.arch == 'VGG16':
            x = vgg_preprocess(x)
            
        elif self.arch == 'ResNet50':
            x = resnet_preprocess(x)
            
        elif self.arch == 'Xception':
            x = xception_preprocess(x)
        
        # Trích xuất đặc trưng
        features = self.model.predict(x)
        
        # Chuẩn hóa (scale) các đặc trưng bằng cách chia cho L2-norm ( độ dài của vector feature)
        features = features / np.linalg.norm(features)
        
        return features