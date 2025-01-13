import torch
from PIL import Image
import numpy as np
from pathlib import Path
import cv2

class ChickenCounter:
    def __init__(self, confidence_threshold=0.3):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.confidence_threshold = confidence_threshold
        self.target_class = 14
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    def enhance_image(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l2 = self.clahe.apply(l)
        lab = cv2.merge((l2,a,b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
        return enhanced
    
    def preprocess_image(self, image_path):
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        enhanced_image = self.enhance_image(image)
        return enhanced_image
    
    def detect_at_multiple_scales(self, image):
        all_predictions = []
        scales = [0.8, 1.0, 1.2]
        original_size = image.shape[:2]
        
        for scale in scales:
            height = int(original_size[0] * scale)
            width = int(original_size[1] * scale)
            scaled_img = cv2.resize(image, (width, height))
            
            results = self.model(scaled_img)
            predictions = results.pred[0]
            
            if len(predictions):
                # Create a copy of the predictions before modifying
                scaled_predictions = predictions.clone()
                scaled_predictions[:, [0, 2]] *= (original_size[1] / width)
                scaled_predictions[:, [1, 3]] *= (original_size[0] / height)
                all_predictions.append(scaled_predictions)
        
        if all_predictions:
            return torch.cat(all_predictions, dim=0)
        return torch.empty((0, 6))
    
    def count_chickens(self, image_path, visualize=False):
        image = self.preprocess_image(image_path)
        predictions = self.detect_at_multiple_scales(image)
        
        # Filter detections
        bird_detections = predictions[
            (predictions[:, -1] == self.target_class) & 
            (predictions[:, 4] >= self.confidence_threshold)
        ]
        
        # Convert to numpy for easier processing
        if len(bird_detections):
            bird_detections = bird_detections.cpu().numpy()
            
            # Apply NMS
            boxes = bird_detections[:, :4]
            scores = bird_detections[:, 4]
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                self.confidence_threshold,
                0.45  # NMS threshold
            )
            if len(indices) > 0:
                bird_detections = bird_detections[indices.flatten()]
            
        chicken_count = len(bird_detections)
        
        if visualize:
            annotated_image = image.copy()
            for detection in bird_detections:
                x1, y1, x2, y2 = detection[:4].astype(int)
                conf = detection[4]
                
                # Draw rectangle
                cv2.rectangle(
                    annotated_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )
                
                # Add confidence score
                label = f'Chicken {conf:.2f}'
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            
            # Add total count to image
            cv2.putText(
                annotated_image,
                f'Total Chickens: {chicken_count}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            return chicken_count, annotated_image
        
        return chicken_count

    def process_video(self, video_path, output_path=None, display=False):
        cap = cv2.VideoCapture(video_path)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            count, annotated_frame = self.count_chickens(frame, visualize=True)
            
            if output_path:
                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                
            if display:
                cv2.imshow('Chicken Counter', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if output_path:
            out.release()
        if display:
            cv2.destroyAllWindows()