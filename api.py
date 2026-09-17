from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
from utils import load_image, load_model, get_grounding_output, postprocess_grounding_output, to_serializable
from utils import person_recognition, flag_logo_recognition, flower_bird_car_airplane_recognition, landmark_recognition
from dinov3_recognition import DinoV3Predictor
import os
import pickle, json
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

app = FastAPI()
models = {}
device = torch.device('cuda:4')
torch.cuda.set_device(device)
task_type_list = ['person', 'landmark', 'flag_logo', 'airplane', 'car', 'bird']
# task_type_list = ['person']
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = Path(__file__).resolve().parent
DINOV3_PREDICTION_ROOT = Path(os.getenv(
    "DINOV3_PREDICTION_ROOT",
    "/home/liujingmeng/flag_logo/dinov3_prediction",
))
DINOV3_DEVICE = os.getenv("DINOV3_DEVICE", str(device))
DINOV3_TOP_K = int(os.getenv("DINOV3_TOP_K", "5"))
DINOV3_CONFIDENCE_THRESHOLD = float(os.getenv("DINOV3_CONFIDENCE_THRESHOLD", "0.7"))
FLAG_LOGO_NMS_IOU_THRESHOLD = float(os.getenv("FLAG_LOGO_NMS_IOU_THRESHOLD", "0.7"))
CAPTION_API_URL = os.getenv("CAPTION_API_URL", "http://127.0.0.1:8000/caption")

def load_grounding_dino():
    print('Start loading GroundingDINO')
    config_file = 'groundingdino/config/GroundingDINO_SwinT_OGC.py'
    checkpoint_path = 'weights/groundingdino_swint_ogc.pth'
    model = load_model(config_file, checkpoint_path, device, cpu_only=False)
    print('GroundingDINO loaded!')
    return model


def load_face_model():
    print('Start loading Face Module.')
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    import numpy as np
    providers = ['CPUExecutionProvider']
    app = FaceAnalysis(name='buffalo_l',providers=providers)
    det_thresh = 0.3
    det_size = (640,640)
    app.prepare(ctx_id=0,det_thresh=det_thresh,det_size=det_size)
    face_gallery_path = 'weights/merged_face_gallery.npz'
    data = np.load(face_gallery_path,allow_pickle=True)
    uids = np.asarray(data['uid']).astype(str)
    names = np.asarray(data['name']).astype(str)
    datasets = np.asarray(data['dataset']).astype(str)
    embs = np.asarray(data['embedding'], dtype=np.float32)
    # l2 norm already normed
    # norm = np.linalg.norm(embs, axis=1, keepdims=True)
    # print('norm::::', norm)
    # norm = np.maximum(norm, 1e-12)
    # embs /= norm
    gallery =  {"uid": uids, "name": names, "dataset": datasets, "embedding": embs}
    return app, gallery
    # from facenet_pytorch import InceptionResnetV1, MTCNN
    # face_resnet = InceptionResnetV1(classify=False,pretrained='vggface2').to(device)
    # face_dict_dir = "weights/face_embeddings_dict.pkl"
    # assert os.path.exists(face_dict_dir)
    # face_resnet.eval()
    # with open(face_dict_dir, 'rb') as file:
    #     face_embeddings_dict = pickle.load(file)
    # mtcnn = MTCNN(image_size=160, margin=14, device=device)
    # print('Face Module loaded!')
    # return face_resnet, mtcnn, face_embeddings_dict


def load_landmark_model():
    print('Start loading Landmark Module.')
    import onnxruntime as rt
    import pandas as pd
    providers = ['CPUExecutionProvider']
    model = rt.InferenceSession("weights/landmark/modified_headlessmodel.onnx", providers=providers)
    model2 = rt.InferenceSession("weights/landmark/modified_headmodel.onnx", providers=providers)
    landmark_classes = pd.read_csv('weights/landmark/class2name.csv', index_col='class_id')
    landmark_classes = landmark_classes['name'].to_dict()
    print('Landmark Module loaded!')    
    return [model, model2], landmark_classes


def load_flower_bird_car_airplane_model(model_type):
    print(f'Start loading {model_type.capitalize()} Module.')    
    import timm
    assert model_type in ['bird', 'flower', 'car', 'airplane']
    num_classes_dict = {'bird': 200, 'flower': 102, 'car': 196, 'airplane': 100}
    checkpoint_path = f"weights/{model_type}_model_best.pth"
    model = timm.create_model(
        "tresnet_l",
        num_classes=num_classes_dict[model_type],
        in_chans=3,
        pretrained=False,
    ).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # 仅限你信任来源
    state_dict = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"{model_type} missing={len(missing)} unexpected={len(unexpected)}")
    class_map = json.load(open(f"weights/{model_type}_class_map.json", 'rb'))
    print(f'{model_type.capitalize()} Module loaded!')
    return model, class_map


def load_flower_bird_car_airplane_transform():
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return transform


def load_flag_logo_model():
    print('Start loading unified DINOv3 flag/logo module.')
    model = DinoV3Predictor(DINOV3_PREDICTION_ROOT, DINOV3_DEVICE)
    print('Unified DINOv3 flag/logo module loaded!')
    return model


def load_all_models():
    model = load_grounding_dino()
    models['detection'] = model
    fine_grained_transforms = load_flower_bird_car_airplane_transform()
    for task in task_type_list:
        if task == 'person':
            models[task] = load_face_model()
        elif task == 'landmark':
            models[task] = load_landmark_model()
        elif task == 'flag_logo':
            models[task] = load_flag_logo_model()
        elif task == 'airplane':
            airplane_model = load_flower_bird_car_airplane_model('airplane')
            models[task] = [fine_grained_transforms] + list(airplane_model)
        elif task == 'car':
            car_model = load_flower_bird_car_airplane_model('car')
            models[task] = [fine_grained_transforms] + list(car_model)
        elif task == 'bird':
            bird_model = load_flower_bird_car_airplane_model('bird')
            models[task] = [fine_grained_transforms] + list(bird_model)
        elif task == 'flower':
            flower_model = load_flower_bird_car_airplane_model('flower')
            models[task] = [fine_grained_transforms] + list(flower_model)


def return_task(task_type, image, fine_grained_list):
    assert task_type in task_type_list 
    if task_type == 'person':
        # face_resnet, mtcnn, face_embeddings_dict = models[task_type]
        app, face_gallery = models[task_type]
        return person_recognition, (app, face_gallery, image, fine_grained_list[task_type])
    elif task_type == 'landmark':
        landmark_model, landmark_classes = models[task_type]
        return landmark_recognition, (landmark_model, landmark_classes, image, fine_grained_list[task_type])
    elif task_type == 'flag_logo':
        return flag_logo_recognition, (
            models[task_type], image, fine_grained_list[task_type],
            DINOV3_TOP_K, DINOV3_CONFIDENCE_THRESHOLD, CAPTION_API_URL,
            FLAG_LOGO_NMS_IOU_THRESHOLD,
        )
    elif task_type == 'airplane':
        fine_grained_transform, airplane_model, airplane_class_map = models[task_type]
        return flower_bird_car_airplane_recognition, ('airplane', airplane_model, fine_grained_transform, airplane_class_map, image, fine_grained_list[task_type], device)
    elif task_type == 'car':
        fine_grained_transform, car_model, car_class_map = models[task_type]
        return flower_bird_car_airplane_recognition, ('car', car_model, fine_grained_transform, car_class_map, image, fine_grained_list[task_type], device)
    elif task_type == 'bird':
        fine_grained_transform, bird_model, bird_class_map = models[task_type]
        return flower_bird_car_airplane_recognition, ('bird', bird_model, fine_grained_transform, bird_class_map, image, fine_grained_list[task_type], device)
    elif task_type == 'flower':
        fine_grained_transform, flower_model, flower_class_map = models[task_type]
        return flower_bird_car_airplane_recognition, ('flower', flower_model, fine_grained_transform, flower_class_map, image, fine_grained_list[task_type], device)
    else:
        assert False, 'Not Implemented!'

        
def fine_grained_recogntion(image, fine_grained_list, results):
    print('Fine-grained Recognition.')
    tasks = []
    for key in task_type_list:
        if len(fine_grained_list[key]) > 0:
            tasks.append(return_task(key, image, fine_grained_list))
    print('task num:', len(tasks))
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(x[0], *x[1]) for x in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print('Task raised an exception:', e)
    return results


def get_matched_task_types(label_name: str):
    """
    Match detection label to enabled task types with token-level exact matching.
    This avoids accidental cross-triggering (e.g. logo triggering flag).
    """
    normalized = label_name.lower().replace("/", " ").replace("-", " ").replace(",", " ")
    tokens = set(normalized.split())
    matched = [task for task in task_type_list if task != 'flag_logo' and task in tokens]
    if 'flag_logo' in task_type_list and ({'flag', 'logo'} & tokens):
        matched.append('flag_logo')
    return matched
        
           
@app.on_event("startup")
async def startup_event():
    load_all_models()


@app.post("/groundingdino/predict")
async def predict(file: UploadFile = File(...), text_prompt: str = Form(...),
                  box_threshold: float = Form(...), text_threshold: float = Form(None),
                  token_spans: str = Form(None)):
    # 把空字符串归一化为 None，避免误入 given-phrase 模式
    if token_spans is not None and token_spans.strip() == "":
        token_spans = None
    # text_threshold 未提供时给个兜底默认值
    if text_threshold is None and token_spans is None:
        text_threshold = 0.25
    image_pil, image = load_image(file.file)
    boxes_filt, pred_phrases = get_grounding_output(
        models['detection'], image, text_prompt, device, box_threshold, text_threshold, token_spans=token_spans
    )
    size = image_pil.size
    boxes_filt, pred_phrases = postprocess_grounding_output(boxes_filt, pred_phrases, size)
    print('detection prompt:', text_prompt, 'detection result:', boxes_filt, pred_phrases)
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    bboxes = [x.tolist() for x in pred_dict['boxes']]
    assert len(bboxes) == len(pred_phrases)
    results = []
    current_id = 1
    fine_grained_list = {t: [] for t in task_type_list}
    for box, label in zip(bboxes, pred_phrases):
        res_item = {}
        name, confidence = label
        assert name != ""
        res_item['bbox_id'] = current_id
        current_id += 1
        res_item['bbox'] = box
        res_item['object_name'] = name
        res_item['probability'] = confidence
        for key in get_matched_task_types(name):
            fine_grained_list[key].append(res_item)
        results.append(res_item)
    results = fine_grained_recogntion(image_pil, fine_grained_list, results)
    print('results:', results)
    return to_serializable(results)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=7579)
