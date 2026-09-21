from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
from utils import load_image, load_model, get_grounding_output, postprocess_grounding_output, get_boxes, parse_label_confidence
from utils import person_recognition, logo_recognition, flower_bird_car_airplane_recognition, landmark_recognition, flag_recognition
import os
import pickle, json
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

app = FastAPI()
device = torch.device('cuda:0')
torch.cuda.set_device(device)

def load_grounding_dino():
    print('Start loading GroundingDINO')
    config_file = 'groundingdino/config/GroundingDINO_SwinT_OGC.py'
    checkpoint_path = 'weights/groundingdino_swint_ogc.pth'
    model = load_model(config_file, checkpoint_path, device, cpu_only=False)
    print('GroundingDINO loaded!')
    return model


def load_face_model():
    print('Start loading Face Module.')
    from facenet_pytorch import InceptionResnetV1, MTCNN
    face_resnet = InceptionResnetV1(classify=False,pretrained='vggface2').to(device)
    face_dict_dir =  "weights/face_embeddings_dict.pkl"
    assert os.path.exists(face_dict_dir)
    face_resnet.eval()
    with open(face_dict_dir, 'rb') as file:
        face_embeddings_dict = pickle.load(file)
    mtcnn = MTCNN(image_size=160, margin=14, device=device)
    print('Face Module loaded!')
    return face_resnet, mtcnn, face_embeddings_dict


def load_landmark_model():
    import onnxruntime as rt
    import pandas as pd
    providers = ['CPUExecutionProvider']
    model = rt.InferenceSession("weights/landmark/modified_headlessmodel.onnx", providers=providers)
    model2 = rt.InferenceSession("weights/landmark/modified_headmodel.onnx", providers=providers)
    landmark_classes = pd.read_csv('weights/landmark/class2name.csv', index_col='class_id')
    landmark_classes = landmark_classes['name'].to_dict()
    return [model, model2], landmark_classes


def load_logo_model():
    import clip
    logo_model, logo_preprocess = clip.load('weights/ViT-B-16.pt',device=device,jit=False)
    checkpoint = torch.load('weights/best_model_vitb16.pt')
    logo_model.load_state_dict(checkpoint['model_state_dict'])
    logo_model.eval()
    with open('weights/logo_class.json') as f:
        logo_class_map = json.load(f)
    logo_classes = []
    logo_names = []
    for v in logo_class_map.values():
        logo_classes.append("a photo of " + v)
        logo_names.append(v)
    logo_classes = clip.tokenize(logo_classes).to(device)
    return logo_model, logo_preprocess, logo_classes, logo_names


def load_flag_model():
    from torchvision import transforms
    image_transforms =transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

    flag_model = torch.load('weights/flag_model_best.pt')
    flag_model = flag_model.to(device)
    flag_model.eval()
    flag_classes = json.load(open('weights/flag_classes.json'))
    name_list, idx_to_class = flag_classes['class_names'], flag_classes['idx_mapping']
    flag_class_mapping = {int(k): name_list[v] for k, v in idx_to_class.items()}
    return image_transforms, flag_model, flag_class_mapping


def load_flower_bird_car_airplane_model(model_type):
    import timm
    assert model_type in ['bird', 'flower', 'car', 'airplane']
    num_classes_dict = {'bird': 200, 'flower': 102, 'car': 196, 'airplane': 100}
    checkpoint_path = f"weights/{model_type}_model_best.pth"
    model = timm.create_model(
        'tresnet_l',
        num_classes=num_classes_dict[model_type],
        in_chans=3,
        pretrained=False,
        checkpoint_path=checkpoint_path).to(device)
    class_map = json.load(open(f"weights/{model_type}_class_map.json", 'rb'))
    return model, class_map


def load_flower_bird_car_airplane_transform():
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return transform


def load_all_models():
    print('in load all models')
    with ProcessPoolExecutor() as executor:
        print('in executor')
        future_detection_model = executor.submit(load_grounding_dino)
        future_face_model = executor.submit(load_face_model)
        future_landmark_model = executor.submit(load_landmark_model)
        future_logo_model = executor.submit(load_logo_model)
        future_flag_model = executor.submit(load_flag_model)
        future_airplane_model = executor.submit(load_flower_bird_car_airplane_model, 'airplane')
        future_car_model = executor.submit(load_flower_bird_car_airplane_model, 'car')
        future_bird_model = executor.submit(load_flower_bird_car_airplane_model, 'bird')
        future_flower_model = executor.submit(load_flower_bird_car_airplane_model, 'flower')
        future_fine_grained_transform = executor.submit(load_flower_bird_car_airplane_transform)
        print('submit done')

    detection_model, face_model, landmark_model, logo_model, flag_model, airplane_model, car_model, bird_model, flower_model, finegrained_transform  = [
        future.result() for future in as_completed(
            [future_detection_model, future_face_model, future_landmark_model, future_logo_model, future_flag_model,
             future_airplane_model, future_car_model, future_bird_model, future_flower_model, future_fine_grained_transform])
    ]
    


# load_all_models()
@app.on_event("startup")
async def startup_event():
    load_all_models()

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=7579)