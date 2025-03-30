import abc
import os
from typing import Optional
import typing

import cv2 # type: ignore
import numpy as np # type: ignore
import torch # type: ignore

from ultralytics import YOLO # type: ignore

from .configs.ViTPose_common import data_cfg
from .sort import Sort
from .vit_models.model import ViTPose
from .vit_utils.inference import draw_bboxes, pad_image
from .vit_utils.top_down_eval import keypoints_from_heatmaps
from .vit_utils.util import dyn_model_import, infer_dataset_by_path
from .vit_utils.visualization import draw_points_and_skeleton, joints_dict

from adaface.inference import load_pretrained_model
from adaface import align
from adaface.similarity_check import find_similar_tests

try:
    import torch_tensorrt # type: ignore
except ModuleNotFoundError:
    pass

try:
    import onnxruntime # type: ignore
except ModuleNotFoundError:
    pass

def is_point_in_box(point, box):
    """
    Check if a point (x, y) is inside a bounding box [x1, y1, x2, y2]
    """
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def to_input(pil_rgb_image):
    if pil_rgb_image is None:
        print("Error: Input image is None")
        return None
        
    np_img = np.array(pil_rgb_image)
    
    # Check array dimensions
    if len(np_img.shape) != 3:
        print(f"Error: Expected 3D array, got shape {np_img.shape}")
        return None
        
    try:
        brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
        tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
        return tensor
    except Exception as e:
        print(f"Error in to_input: {str(e)}")
        return None


__all__ = ['VitInference']
np.bool = np.bool_
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


DETC_TO_YOLO_YOLOC = {
    'human': [0],
    'cat': [15],
    'dog': [16],
    'horse': [17],
    'sheep': [18],
    'cow': [19],
    'elephant': [20],
    'bear': [21],
    'zebra': [22],
    'giraffe': [23],
    'animals': [15, 16, 17, 18, 19, 20, 21, 22, 23]
}


class VitInference:
    """
    Class for performing inference using ViTPose models with YOLOv8 human detection and SORT tracking.

    Args:
        model (str): Path to the ViT model file (.pth, .onnx, .engine).
        yolo (str): Path of the YOLOv8 model to load.
        model_name (str, optional): Name of the ViT model architecture to use.
                                    Valid values are 's', 'b', 'l', 'h'.
                                    Defaults to None, is necessary when using .pth checkpoints.
        det_class (str, optional): the detection class. if None it is inferred by the dataset.
                                   valid values are 'human', 'cat', 'dog', 'horse', 'sheep',
                                                    'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                                                    'animals' (which is all previous but human)
        dataset (str, optional): Name of the dataset. If None it's extracted from the file name.
                                 Valid values are 'coco', 'coco_25', 'wholebody', 'mpii',
                                                  'ap10k', 'apt36k', 'aic'
        yolo_size (int, optional): Size of the input image for YOLOv8 model. Defaults to 320.
        device (str, optional): Device to use for inference. Defaults to 'cuda' if available, else 'cpu'.
        is_video (bool, optional): Flag indicating if the input is video. Defaults to False.
        single_pose (bool, optional): Flag indicating if the video (on images this flag has no effect)
                                      will contain a single pose.
                                      In this case the SORT tracker is not used (increasing performance)
                                      but people id tracking
                                      won't be consistent among frames.
        yolo_step (int, optional): The tracker can be used to predict the bboxes instead of yolo for performance,
                                   this flag specifies how often yolo is applied (e.g. 1 applies yolo every frame).
                                   This does not have any effect when is_video is False.
    """

    def __init__(self, model: str,
                 yolo: str,
                 yolo_face: str,
                 model_name: Optional[str] = None,
                 det_class: Optional[str] = None,
                 dataset: Optional[str] = None,
                 yolo_size: Optional[int] = 320,
                 device: Optional[str] = None,
                 is_video: Optional[bool] = False,
                 single_pose: Optional[bool] = False,
                 yolo_step: Optional[int] = 1):
        assert os.path.isfile(model), f'The model file {model} does not exist'
        assert os.path.isfile(yolo), f'The YOLOv8 model {yolo} does not exist'

        # Device priority is cuda / mps / cpu
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device
        self.yolo = YOLO(yolo, task='detect')
        self.yolo_face = YOLO(yolo_face, task='detect')

        self.ada_face = load_pretrained_model('ir_50')
        feature, norm = self.ada_face(torch.randn(2,3,112,112))

        self.yolo_size = yolo_size
        self.yolo_step = yolo_step
        self.is_video = is_video
        self.single_pose = single_pose
        self.reset()

        # State saving during inference
        self.save_state = True  # Can be disabled manually
        self._img = None
        self._yolo_res = None
        self._yolo_res_face = None
        self.identities = None
        self._tracker_res = None
        self._keypoints = None
        self.identity_map = {}

        # Use extension to decide which kind of model has been loaded
        use_onnx = model.endswith('.onnx')
        use_trt = model.endswith('.engine')


        # Extract dataset name
        if dataset is None:
            dataset = infer_dataset_by_path(model)

        assert dataset in ['mpii', 'coco', 'coco_25', 'wholebody', 'aic', 'ap10k', 'apt36k'], \
            'The specified dataset is not valid'

        # Dataset can now be set for visualization
        self.dataset = dataset

        # if we picked the dataset switch to correct yolo classes if not set
        if det_class is None:
            det_class = 'animals' if dataset in ['ap10k', 'apt36k'] else 'human'
        self.yolo_classes = DETC_TO_YOLO_YOLOC[det_class]

        assert model_name in [None, 's', 'b', 'l', 'h'], \
            f'The model name {model_name} is not valid'

        # onnx / trt models do not require model_cfg specification
        if model_name is None:
            assert use_onnx or use_trt, \
                'Specify the model_name if not using onnx / trt'
        else:
            # Dynamically import the model class
            model_cfg = dyn_model_import(self.dataset, model_name)

        self.target_size = data_cfg['image_size']
        if use_onnx:
            self._ort_session = onnxruntime.InferenceSession(model,
                                                             providers=['CUDAExecutionProvider',
                                                                        'CPUExecutionProvider'])
            inf_fn = self._inference_onnx
        else:
            self._vit_pose = ViTPose(model_cfg)
            self._vit_pose.eval()

            if use_trt:
                self._vit_pose = torch.jit.load(model)
            else:
                ckpt = torch.load(model, map_location='cpu', weights_only=True)
                if 'state_dict' in ckpt:
                    self._vit_pose.load_state_dict(ckpt['state_dict'])
                else:
                    self._vit_pose.load_state_dict(ckpt)
                self._vit_pose.to(torch.device(device))

            inf_fn = self._inference_torch

        # Override _inference abstract with selected engine
        self._inference = inf_fn  # type: ignore

    def reset(self):
        """
        Reset the inference class to be ready for a new video.
        This will reset the internal counter of frames, on videos
        this is necessary to reset the tracker.
        """
        min_hits = 3 if self.yolo_step == 1 else 1
        use_tracker = self.is_video and not self.single_pose
        self.tracker = Sort(max_age=self.yolo_step,
                            min_hits=min_hits,
                            iou_threshold=0.3) if use_tracker else None  # TODO: Params
        self.frame_counter = 0

    @classmethod
    def postprocess(cls, heatmaps, org_w, org_h):
        """
        Postprocess the heatmaps to obtain keypoints and their probabilities.

        Args:
            heatmaps (ndarray): Heatmap predictions from the model.
            org_w (int): Original width of the image.
            org_h (int): Original height of the image.

        Returns:
            ndarray: Processed keypoints with probabilities.
        """
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                               center=np.array([[org_w // 2,
                                                                 org_h // 2]]),
                                               scale=np.array([[org_w, org_h]]),
                                               unbiased=True, use_udp=True)
        return np.concatenate([points[:, :, ::-1], prob], axis=2)

    @abc.abstractmethod
    def _inference(self, img: np.ndarray) -> np.ndarray:
        """
        Abstract method for performing inference on an image.
        It is overloaded by each inference engine.

        Args:
            img (ndarray): Input image for inference.

        Returns:
            ndarray: Inference results.
        """
        raise NotImplementedError

    def inference(self, img: np.ndarray) -> dict[typing.Any, typing.Any]:
        """
        Perform inference on the input image.

        Args:
            img (ndarray): Input image for inference in RGB format.

        Returns:
            dict[typing.Any, typing.Any]: Inference results.
        """

        # First use YOLOv8 for detection
        res_pd = np.empty((0, 5))
        results = None
        face_results = None
        if (self.tracker is None or
           (self.frame_counter % self.yolo_step == 0 or self.frame_counter < 3)):
            results = self.yolo(img[..., ::-1], verbose=False, imgsz=self.yolo_size,
                                device=self.device if self.device != 'cuda' else 0,
                                classes=self.yolo_classes)[0]
            face_results = self.yolo_face(img[..., ::-1], verbose=False, imgsz=self.yolo_size,
                                device=self.device if self.device != 'cuda' else 0)[0]
            res_pd = np.array([r[:5].tolist() for r in  # TODO: Confidence threshold
                               results.boxes.data.cpu().numpy() if r[4] > 0.35]).reshape((-1, 5))
        self.frame_counter += 1


        # Easy VitPose vanilla code:

        frame_keypoints = {}
        scores_bbox = {}
        ids = None
        if self.tracker is not None:
            res_pd = self.tracker.update(res_pd)
            ids = res_pd[:, 5].astype(int).tolist()

        # Prepare boxes for inference
        bboxes = res_pd[:, :4].round().astype(int)
        scores = res_pd[:, 4].tolist()
        pad_bbox = 10

        
        # END Easy VitPose vanilla code:

        # Run adaface and find appropriate faces

        current_dir = os.path.dirname(os.path.abspath(__file__))
        subject_face_path = os.path.join(current_dir, "faces")

        new_face_results = []

        subject_count = len(os.listdir(subject_face_path))
        in_frame_count = 0
        if face_results is not None and \
            face_results.boxes is not None and \
            face_results.boxes.data is not None and \
            len(face_results.boxes.data) > 0:

            for detection in face_results.boxes.data:
                x1, y1, x2, y2, conf, class_id = detection
                
                # Convert to integers for slicing
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                face_center_x = (x1 + x2) / 2
                face_center_y = (y1 + y2) / 2
                face_center = (face_center_x, face_center_y)
                
                if self.identity_map.get(ids[index]) is not None and is_point_in_box(face_center, bbox):
                    # We already have a subject asociated with this bb
                    pass
                else:
                    in_frame_count += 1
                    new_face_results.append((x1, y1, x2, y2, conf, class_id))

        total_people = subject_count + in_frame_count

        print("total_people:", total_people)

        features = []
        filenames = []
        identities = {}
        idnetity_map = self.identity_map

        # Process each subject
        for fname in sorted(os.listdir(subject_face_path)):
            path = os.path.join(subject_face_path, fname)
            aligned_rgb_img = align.get_aligned_face(image_path=path)
            bgr_tensor_input = to_input(aligned_rgb_img)
            feature, _ = self.ada_face(bgr_tensor_input)
            features.append(feature)
            filenames.append(fname)

        # Process each detected face
        if len(new_face_results) > 0:
            for detection in new_face_results:
                x1, y1, x2, y2, conf, class_id = detection
                print("Detection:", detection)
                print("Type:", type(detection))
                
                # Convert to integers for slicing
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                if x1 >= x2 or y1 >= y2:
                    print(f"Invalid coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                    continue
                
                # Extract face region from original image
                face_crop = img[y1:y2, x1:x2]
                
                # Align the face (assuming align.get_aligned_face can work with numpy arrays)
                aligned_rgb_img = align.get_aligned_face(np_array=face_crop)
                
                # Process the aligned face
                try:
                    bgr_tensor_input = to_input(aligned_rgb_img)
                    feature, _ = self.ada_face(bgr_tensor_input)
                    features.append(feature)
                except Exception as e:
                    print("Error:", e)
                    pass

            similarity_matrix = torch.cat(features) @ torch.cat(features).T
            print(similarity_matrix)

            # For each person in results, check if the person has a face that matches AdaFace 

            similar_tests = find_similar_tests(similarity_matrix, threshold=0.3, num_examples=subject_count)

            for frame_index, source_index, score in similar_tests:
                
                face_center_x = (x1 + x2) / 2
                face_center_y = (y1 + y2) / 2
                face_center = (face_center_x, face_center_y)

                identities[x1] = (filenames[source_index].split(".")[0], face_center)

        # Easy VitPose vanilla code:

        if ids is None:
            ids = range(len(bboxes))

        for index, (bbox, id, score) in enumerate(zip(bboxes, ids, scores)):
            # TODO: Slightly bigger bbox
            bbox[[0, 2]] = np.clip(bbox[[0, 2]] + [-pad_bbox, pad_bbox], 0, img.shape[1])
            bbox[[1, 3]] = np.clip(bbox[[1, 3]] + [-pad_bbox, pad_bbox], 0, img.shape[0])

            # Crop image and pad to 3/4 aspect ratio
            img_inf = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            img_inf, (left_pad, top_pad) = pad_image(img_inf, 3 / 4)

            keypoints = self._inference(img_inf)[0]
            # Transform keypoints to original image
            keypoints[:, :2] += bbox[:2][::-1] - [top_pad, left_pad]
            frame_keypoints[id] = keypoints
            scores_bbox[id] = score  # Replace this with avg_keypoint_conf*person_obj_conf. For now, only person_obj_conf from yolo is being used.

            for x1, body in identities.items():
                name, center = body
                print("name:", name)
                print("center:", center)
                if is_point_in_box(center, bbox):
                    idnetity_map[id] = name
                    # ids[index] = name

        if self.save_state:
            self._img = img
            self._yolo_res = results
            self._yolo_res_face = face_results
            self._tracker_res = (bboxes, ids, scores)
            self._keypoints = frame_keypoints
            self._scores_bbox = scores_bbox
            self.identities = identities
            self.identity_map = idnetity_map

        return frame_keypoints

    def draw(self, show_yolo=True, show_raw_yolo=False, confidence_threshold=0.5):
        """
        Draw keypoints and bounding boxes on the image.

        Args:
            show_yolo (bool, optional): Whether to show YOLOv8 bounding boxes. Default is True.
            show_raw_yolo (bool, optional): Whether to show raw YOLOv8 bounding boxes. Default is False.

        Returns:
            ndarray: Image with keypoints and bounding boxes drawn.
        """
        img = self._img.copy()
        bboxes, ids, scores = self._tracker_res

        if self._yolo_res is not None and (show_raw_yolo or (self.tracker is None and show_yolo)):
            img = np.array(self._yolo_res.plot())[..., ::-1]

        if self._yolo_res_face is not None:
            # Create a copy of the image
            img_with_boxes = img.copy()
            
            # Get the boxes
            boxes = self._yolo_res_face.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]

                x1_alt = int(box.data[0][0])

                if self.identities.get(x1_alt) is not None:
                    
                    # Draw box
                    cv2.rectangle(img_with_boxes, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 255, 0), 2)
                    
                    # Add custom label
                    custom_label = self.identities.get(x1_alt)[0]
                    cv2.putText(img_with_boxes, 
                                custom_label, 
                                (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.75, 
                                (0, 255, 0), 
                                2)
            
            img = img_with_boxes

        if show_yolo and self.tracker is not None:
            img = draw_bboxes(img, bboxes, ids, scores, self.identity_map) # HERE

        img = np.array(img)[..., ::-1]  # RGB to BGR for cv2 modules
        for idx, k in self._keypoints.items():
            img = draw_points_and_skeleton(img.copy(), k,
                                           joints_dict()[self.dataset]['skeleton'],
                                           person_index=idx,
                                           points_color_palette='gist_rainbow',
                                           skeleton_color_palette='jet',
                                           points_palette_samples=10,
                                           confidence_threshold=confidence_threshold)
        return img[..., ::-1]  # Return RGB as original

    def pre_img(self, img):
        org_h, org_w = img.shape[:2]
        img_input = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR) / 255
        img_input = ((img_input - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        return img_input, org_h, org_w

    @torch.no_grad()
    def _inference_torch(self, img: np.ndarray) -> np.ndarray:
        # Prepare input data
        img_input, org_h, org_w = self.pre_img(img)
        img_input = torch.from_numpy(img_input).to(torch.device(self.device))

        # Feed to model
        heatmaps = self._vit_pose(img_input).detach().cpu().numpy()
        return self.postprocess(heatmaps, org_w, org_h)

    def _inference_onnx(self, img: np.ndarray) -> np.ndarray:
        # Prepare input data
        img_input, org_h, org_w = self.pre_img(img)

        # Feed to model
        ort_inputs = {self._ort_session.get_inputs()[0].name: img_input}
        heatmaps = self._ort_session.run(None, ort_inputs)[0]
        return self.postprocess(heatmaps, org_w, org_h)