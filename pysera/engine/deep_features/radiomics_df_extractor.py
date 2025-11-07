import gc
import logging
from typing import Optional

import numpy as np

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch torchvision")


class MedicalImageDataset(Dataset):
    """Custom dataset for medical images with ROI extraction."""

    def __init__(self, image_data, mask_data, transform=None):
        self.image_data = image_data
        self.mask_data = mask_data
        self.transform = transform

    def __len__(self):
        return 1  # Single image-mask pair

    def __getitem__(self, idx):
        image = self.image_data.astype(np.float32)
        mask = self.mask_data.astype(np.float32)

        # --- Apply mask to extract ROI ---
        roi_image = image * (mask > 0)

        # --- Normalize to [0, 1] ---
        vmin, vmax = np.min(roi_image), np.max(roi_image)
        if vmax > vmin:
            roi_image = (roi_image - vmin) / (vmax - vmin)
        else:
            roi_image = np.zeros_like(roi_image, dtype=np.float32)

        # --- Select the slice with the largest ROI area instead of middle slice ---
        if len(roi_image.shape) == 3:
            mask_sum = mask.sum(axis=(0, 1))
            best_slice = int(np.argmax(mask_sum))
            roi_image = roi_image[:, :, best_slice]

        # --- Convert to RGB (3 channels) ---
        roi_image_rgb = np.stack([roi_image] * 3, axis=0)  # (3, H, W)
        roi_tensor = torch.from_numpy(roi_image_rgb).float().unsqueeze(0)  # add batch dim temporarily for resize

        # --- Resize safely using torch interpolate ---
        roi_tensor = F.interpolate(roi_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        roi_tensor = roi_tensor.squeeze(0)  # remove batch dimension (C, H, W)

        # --- Normalize with ImageNet stats ---
        if self.transform:
            roi_tensor = self.transform(roi_tensor)

        return roi_tensor


def _extract_features_from_model(model, model_name, image_tensor):
    """Extract features from a specific CNN model."""
    if model is None:
        return {}
    try:
        with torch.no_grad():
            if hasattr(model, 'classifier') and 'VGG' in str(type(model)):
                features = model.features(image_tensor)
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            elif hasattr(model, 'fc'):
                features = model.conv1(image_tensor)
                features = model.bn1(features)
                features = model.relu(features)
                features = model.maxpool(features)
                features = model.layer1(features)
                features = model.layer2(features)
                features = model.layer3(features)
                features = model.layer4(features)
                features = model.avgpool(features)
                features = features.view(features.size(0), -1)
            elif hasattr(model, 'features') and 'DenseNet' in str(type(model)):
                features = model.features(image_tensor)
                features = torch.nn.functional.relu(features, inplace=True)
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            else:
                features = model(image_tensor)

            features_np = features.cpu().numpy().flatten()
            feature_names = [f"{model_name}_feature_{i}" for i in range(len(features_np))]
            return dict(zip(feature_names, features_np))
    except Exception as e:
        logging.error(f"Error extracting features from {model_name}: {e}")
        return {}


class DeepLearningRadiomicsProcessor:
    """Deep Learning radiomics processor for ViSERA."""

    def __init__(self, selected_model: str = "resnet50"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep learning radiomics")

        valid_models = {"all", "resnet50", "vgg16", "densenet121"}
        if selected_model.lower() not in valid_models:
            raise ValueError(f"Invalid selected_model '{selected_model}'. Choose from {valid_models}.")

        self.selected_model = selected_model.lower()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        self._initialize_models()

    def _initialize_models(self):
        """Initialize pre-trained CNN models for feature extraction."""
        # Only normalization transform; resize now done manually
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        def _safe_load(model_func, weights, model_name):
            try:
                model = model_func(weights=weights)
                model.eval()
                model.to(self.device)
                logging.info(f"{model_name} loaded successfully")
                return model
            except Exception as e:
                logging.warning(f"Failed to load {model_name}: {e}. Using random weights.")
                model = model_func(weights=None)
                model.eval()
                model.to(self.device)
                return model

        if self.selected_model in {"resnet50", "all"}:
            self.resnet50 = _safe_load(models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, "ResNet50")
        else:
            self.resnet50 = None

        if self.selected_model in {"vgg16", "all"}:
            self.vgg16 = _safe_load(models.vgg16, models.VGG16_Weights.IMAGENET1K_V1, "VGG16")
        else:
            self.vgg16 = None

        if self.selected_model in {"densenet121", "all"}:
            self.densenet121 = _safe_load(models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1, "DenseNet121")
        else:
            self.densenet121 = None

    def _extract_deep_features(self, image_data, mask_data, patient_id, roi_name):
        try:
            dataset = MedicalImageDataset(image_data, mask_data, self.transform)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            all_features = {}

            for _, image_tensor in enumerate(dataloader):
                image_tensor = image_tensor.to(self.device)

                if self.selected_model in {"resnet50", "all"}:
                    all_features.update(_extract_features_from_model(self.resnet50, "ResNet50", image_tensor))
                if self.selected_model in {"vgg16", "all"}:
                    all_features.update(_extract_features_from_model(self.vgg16, "VGG16", image_tensor))
                if self.selected_model in {"densenet121", "all"}:
                    all_features.update(_extract_features_from_model(self.densenet121, "DenseNet121", image_tensor))

                logging.info(f"[%s] roi_name: '%s' Extracted %d deep learning features",
                             patient_id, roi_name, len(all_features))

            return all_features
        except Exception as e:
            logging.error(f"Error in deep feature extraction: {e}")
            return {}

    def process_single_image_pair(self, image_array: np.ndarray, mask_array: np.ndarray,
                                  image_name: Optional[str] = None, roi_name: Optional[str] = None):
        try:
            patient_id = image_name or "unknown_patient"
            logging.info(f"[%s] roi_name: '%s' Processing deep learning radiomics using %s",
                         patient_id, roi_name, self.selected_model)

            deep_features = self._extract_deep_features(image_array, mask_array, patient_id, roi_name)
            if not deep_features:
                logging.error(f"[%s] roi_name: '%s' : No deep learning features extracted",
                              patient_id, roi_name)
                return None
            return deep_features
        except Exception as e:
            logging.error(f"Error in deep learning processing for {image_name}: {e}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
            return None
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def deep_learning_feature_extractor(selected_model: str = "resnet50"):
    if not TORCH_AVAILABLE:
        logging.error("PyTorch is required for deep learning radiomics")
        return None
    return DeepLearningRadiomicsProcessor(selected_model=selected_model)
