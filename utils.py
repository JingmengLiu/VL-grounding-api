from PIL import Image
from groundingdino.datasets import transforms as T
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
import torch
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
import numpy as np
import math
import re
import urllib.parse
import urllib.request
import urllib.error
import io
import uuid
import json
from typing import Optional
from torchvision.ops import nms
import cv2



def to_serializable(obj):
    """Recursively convert common ML/Numpy/Torch types into JSON-serializable Python types."""
    # torch.Tensor -> python scalar or nested lists
    if isinstance(obj, torch.Tensor):
        obj = obj.detach()
        if obj.numel() == 1:
            return obj.item()
        return obj.cpu().tolist()

    # numpy scalars / arrays
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # containers
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]

    # pathlib.Path
    try:
        from pathlib import Path

        if isinstance(obj, Path):
            return str(obj)
    except Exception:
        pass

    # basic python types
    return obj


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # print('result:', load_res)
    _ = model.eval()
    model = model.to(device)
    return model


def get_grounding_output(model, image, caption, device, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):            
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    # print('caption:', caption)
    # print('caption:', caption)
    if not caption.endswith("."):
        caption = caption + "."
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(caption),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases


def postprocess_grounding_output(boxes, pred_phrases, size):
    pred_phrases = [parse_label_confidence(x) for x in pred_phrases]
    # print('xywh', boxes)
    H, W = size[1], size[0]
    boxes = boxes * torch.Tensor([W, H, W, H])
    # from xywh to xyxy
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    # print('xyxy:', boxes)
    filtered_boxes = []
    filtered_phrases = []
    for box, pred in zip(boxes, pred_phrases):
        if pred[0]:
            filtered_boxes.append(box)
            filtered_phrases.append(pred)
    filtered_boxes = torch.stack(filtered_boxes) if filtered_boxes else torch.empty(0, 4)
    confidences = torch.tensor([x[1] for x in filtered_phrases])
    nms_indices = nms(filtered_boxes, confidences, iou_threshold=0.5)   
    nms_boxes = filtered_boxes[nms_indices]
    nms_phrases = [filtered_phrases[i] for i in nms_indices]
    landmark_boxes = []
    landmark_confidences = []
    for i, pred in enumerate(nms_phrases):
        if pred[0] == "landmark":
            landmark_boxes.append(nms_boxes[i])
            landmark_confidences.append(pred[1])
    if len(landmark_boxes) > 1:
        landmark_boxes = torch.stack(landmark_boxes)
        x_min, y_min = torch.min(landmark_boxes[:, :2], dim=0)[0]
        x_max, y_max = torch.max(landmark_boxes[:, 2:], dim=0)[0]
        merged_landmark_box = torch.tensor([x_min, y_min, x_max, y_max])
        max_confidence = max(landmark_confidences)
        final_boxes = []
        final_phrases = []
        for i, pred in enumerate(nms_phrases):
            if pred[0] != "landmark":
                final_boxes.append(nms_boxes[i])
                final_phrases.append(pred)
        final_boxes.append(merged_landmark_box)
        final_phrases.append(("landmark", max_confidence))
        nms_boxes = torch.stack(final_boxes)
        nms_phrases = final_phrases

    return nms_boxes, nms_phrases


def get_boxes(tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    # draw boxes and masks
    new_boxes = []
    for box, label in zip(boxes, labels):
        # print('before box', box)
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # print('after box', box)        
        new_boxes.append(box)
    tgt['boxes'] = new_boxes

    return tgt


def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # 基于余弦相似度的距离
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric
    return dist


def parse_label_confidence(s):
    match = re.match(r'(\w+)\(([\d.]+)\)', s)
    if match:
        label = match.group(1)
        confidence = float(match.group(2))
        return label, confidence
    else:
        return '', 0

def crop_image(image, xyxys):
    print('crop bbox:',  xyxys)
    x1, y1, x2, y2 = xyxys
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return image.crop((x1, y1, x2, y2))  


def _resize_pil(image_pil, resize_size=(224, 224)):
    if resize_size is None:
        return image_pil
    size = (int(resize_size[0]), int(resize_size[1]))
    if image_pil.size != size:
        return image_pil.resize(size, Image.BICUBIC)
    return image_pil


def _extract_text_from_caption_response(resp_bytes: bytes, content_type: Optional[str]):
    text = resp_bytes.decode("utf-8", errors="ignore").strip()
    if content_type and "application/json" in content_type.lower():
        try:
            obj = json.loads(text)
            # common keys
            for k in ("text", "caption", "result", "answer", "output"):
                if isinstance(obj, dict) and k in obj and isinstance(obj[k], str):
                    return obj[k].strip()
            # nested
            if isinstance(obj, dict):
                for k in ("data", "response"):
                    if k in obj and isinstance(obj[k], dict):
                        for kk in ("text", "caption", "result", "answer"):
                            if kk in obj[k] and isinstance(obj[k][kk], str):
                                return obj[k][kk].strip()
        except Exception:
            return text
    return text


def call_caption_api(image_pil, prompt: str, url: str, timeout: float = 30.0):
    """Call external caption API (started via uvicorn caption:app --port 7578).

    Assumes a multipart/form-data POST with fields:
      - image: uploaded file
      - prompt: text prompt

    Response can be plain text or JSON; this function tries to extract a text field.
    """
    boundary = f"----Boundary{uuid.uuid4().hex}"

    buf = io.BytesIO()
    image_pil.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    def _w(s: str):
        return s.encode("utf-8")

    body = b"".join(
        [
            _w(f"--{boundary}\r\n"),
            _w('Content-Disposition: form-data; name="prompt"\r\n\r\n'),
            _w(prompt),
            _w("\r\n"),
            _w(f"--{boundary}\r\n"),
            _w('Content-Disposition: form-data; name="image"; filename="image.png"\r\n'),
            _w("Content-Type: image/png\r\n\r\n"),
            img_bytes,
            _w("\r\n"),
            _w(f"--{boundary}--\r\n"),
        ]
    )

    req = urllib.request.Request(
        url=url,
        method="POST",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json, text/plain, */*",
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp_bytes = resp.read()
        content_type = resp.headers.get("Content-Type")
        return _extract_text_from_caption_response(resp_bytes, content_type)

def person_recognition(app, face_gallery, image, target):
    print('Start recognizing Person.')
    print('image,', image)
    print('target,', target)
    gallery_embs = face_gallery["embedding"] #[N,512]
    for item in target:
        item_image = crop_image(image, item['bbox'])
        print('item image:', item_image)
        # item_image = np.array(item_image)
        item_image = cv2.cvtColor(np.array(item_image), cv2.COLOR_RGB2BGR)
        try:
            faces = app.get(item_image)
            print("faces:", len(faces) if faces else 0)
        except Exception as e:
            print("app.get error:", repr(e))
            continue
        if not faces:
            print(f"no faces for bbox: {item['bbox']}")
            continue
        if len(faces) > 1:
            max_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        else:
            max_face = faces[0]
        q_emb = max_face.embedding.astype(np.float32)
        q_emb /= np.linalg.norm(q_emb)
        sims = gallery_embs @ q_emb  # [N]
        idx = np.argmax(sims)
        if sims[idx] >= 0.4:
            face_name, face_dataset, face_id = str(face_gallery["name"][idx]), str(face_gallery["dataset"][idx]), str(face_gallery["uid"][idx])
            face_confidence = float(sims[idx])
            item['object_finegrained_name'], item['object_finegrained_dataset'], item['object_finegrained_id'] = face_name, face_dataset, face_id
            item['person'] = {'name': face_name, 'confidence': face_confidence}
        # img_embedding = resnet(img_cropped.unsqueeze(0).to(device)).detach().to('cpu').numpy()
        # distances = []
        # src_img_paths = []
        # for src_img_path, src_img_embedding in embedding_dict.items():
        #     distances.append(distance(src_img_embedding, img_embedding, distance_metric=0)[0])
        #     src_img_paths.append(src_img_path)
        # if np.min(distances) <= 0.6:
        #     most_similar_idx = np.argmin(distances)
        #     face_name = src_img_paths[most_similar_idx].split('/')[-1][:-9].replace("_", " ")
        #     face_confidence = float(1 - np.min(distances))
        #     item['object_finegrained_name'] = face_name
        #     item['person'] = {'name': face_name, 'confidence': face_confidence}
    print('Person Recognition Done!')
    return to_serializable(target)

def person_recognition_old(resnet, mtcnn, embedding_dict, image, target, device):
    print('Start recognizing Person.')
    print('image,', image)
    print('target,', target)
    for item in target:
        item_image = crop_image(image, item['bbox'])
        print('crop?', item_image)
        try:
            img_cropped = mtcnn(item_image)
        except Exception as e:
            print('mtcnn failed:', e)
            continue
        if img_cropped is None:
            print(f"mtcnn failed to crop image for bbox: {item['bbox']}")
            continue
        img_embedding = resnet(img_cropped.unsqueeze(0).to(device)).detach().to('cpu').numpy()
        distances = []
        src_img_paths = []
        for src_img_path, src_img_embedding in embedding_dict.items():
            distances.append(distance(src_img_embedding, img_embedding, distance_metric=0)[0])
            src_img_paths.append(src_img_path)
        if np.min(distances) <= 0.6:
            most_similar_idx = np.argmin(distances)
            face_name = src_img_paths[most_similar_idx].split('/')[-1][:-9].replace("_", " ")
            face_confidence = float(1 - np.min(distances))
            item['object_finegrained_name'] = face_name
            item['person'] = {'name': face_name, 'confidence': face_confidence}
    print('Person Recognition Done!')
    return to_serializable(target)
 

def logo_recognition(processor, model, idx_to_name, image, target, device, threshold=0.7, caption_api_url=None):
    print('Start recognizing Logo (SigLIP2 + caption-api fallback).')
    if len(target) == 0:
        return to_serializable(target)

    prompt = '请判断这是什么公司或组织，只输出英文名。'

    for item in target:
        item_image = crop_image(image, item['bbox'])
        item_image = _resize_pil(item_image, resize_size=(224, 224))

        inputs = processor(images=[item_image], return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(device)

        with torch.no_grad():
            logits, _ = model(pixel_values=pixel_values)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            conf_val = float(conf.item())
            pred_idx = int(pred.item())

        if conf_val >= float(threshold):
            logo_name = idx_to_name.get(pred_idx, f"Unknown_{pred_idx}")
            item['object_finegrained_name'] = logo_name
            item['logo'] = {'name': logo_name, 'confidence': conf_val}
        else:
            if not caption_api_url:
                logo_name = idx_to_name.get(pred_idx, f"Unknown_{pred_idx}")
                item['object_finegrained_name'] = logo_name
                item['logo'] = {'name': logo_name, 'confidence': conf_val}
            else:
                try:
                    pred_text = call_caption_api(item_image, prompt=prompt, url=caption_api_url)
                    pred_text = (pred_text or '').strip()
                except Exception as e:
                    print('caption-api failed:', e)
                    pred_text = ''

                if pred_text:
                    item['object_finegrained_name'] = pred_text
                    # Keep confidence numeric for frontend compatibility.
                    item['logo'] = {'name': pred_text, 'confidence': conf_val}
                else:
                    logo_name = idx_to_name.get(pred_idx, f"Unknown_{pred_idx}")
                    item['object_finegrained_name'] = logo_name
                    item['logo'] = {'name': logo_name, 'confidence': conf_val}

    print('Logo Recognition Done!')
    return to_serializable(target)



def flag_recognition(processor, model, idx_to_name, image, target, device, threshold=0.7, caption_api_url=None):
    print('Start recognizing Flag (SigLIP2 + caption-api fallback).')
    if len(target) == 0:
        return to_serializable(target)

    prompt = '请判断这是什么国家的旗帜，只输出国家英文名。'

    for item in target:
        item_image = crop_image(image, item['bbox'])
        item_image = _resize_pil(item_image, resize_size=(224, 224))

        inputs = processor(images=[item_image], return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(device)

        with torch.no_grad():
            logits, _ = model(pixel_values=pixel_values)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            conf_val = float(conf.item())
            pred_idx = int(pred.item())

        if conf_val >= float(threshold):
            flag_name = idx_to_name.get(pred_idx, f"Unknown_{pred_idx}")
            item['object_finegrained_name'] = flag_name
            item['flag'] = {'name': flag_name, 'confidence': conf_val}
        else:
            if not caption_api_url:
                flag_name = idx_to_name.get(pred_idx, f"Unknown_{pred_idx}")
                item['object_finegrained_name'] = flag_name
                item['flag'] = {'name': flag_name, 'confidence': conf_val}
            else:
                try:
                    pred_text = call_caption_api(item_image, prompt=prompt, url=caption_api_url)
                    pred_text = (pred_text or '').strip()
                except Exception as e:
                    print('caption-api failed:', e)
                    pred_text = ''

                if pred_text:
                    item['object_finegrained_name'] = pred_text
                    # Keep confidence numeric for frontend compatibility.
                    item['flag'] = {'name': pred_text, 'confidence': conf_val}
                else:
                    flag_name = idx_to_name.get(pred_idx, f"Unknown_{pred_idx}")
                    item['object_finegrained_name'] = flag_name
                    item['flag'] = {'name': flag_name, 'confidence': conf_val}

    print('Flag Recognition Done!')
    return to_serializable(target)
        

def flower_bird_car_airplane_recognition(model_type, model, data_transform, class_map, image, target, device):
    print(f'Start recognizing {model_type.capitalize()}.')
    assert model_type in ['bird', 'flower', 'car', 'airplane']
    item_images = []
    for item in target:
        # item_image = image
        item_image = crop_image(image, item['bbox'])
        item_image = data_transform(item_image).unsqueeze(0).to(device)
        item_images.append(item_image)
    item_images = torch.cat(item_images)
    with torch.no_grad():
        output = model(item_images)
        probability = torch.nn.functional.softmax(output, dim=1)
    scores, predicted = torch.max(probability, 1)
    scores = scores.to('cpu').numpy()
    predicted = predicted.to('cpu').numpy()

    for i, item in enumerate(target):
        class_name = class_map[str(predicted[i])]
        class_confidence = float(scores[i])
        # print('name:', class_name, 'class_conf:', class_confidence)
        item['object_finegrained_name'] = class_name
        item[model_type] = {'name': class_name, 'confidence': class_confidence}
    print(f'{model_type.capitalize()} Recognition Done!')
    return to_serializable(target)



def landmark_recognition(model, classes, image, target):
    print('Start recognizing Landmark.')
    def preprocess_image(image):
        image = image.resize((768, 768))
        image = np.array(image).astype(np.float32)
        image = image / 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        return image
    
    def recognition(image):
        feature = model[0].run(None, {"x.1": image})
        output = model[1].run(None, {"features": feature[0]})
        landmark_name = urllib.parse.unquote(classes[output[1][0]])
        landmark_confidence = float(1 / (1 + np.exp(-output[0][0] * 1e5)))
        return landmark_name, landmark_confidence
    
    assert len(target) == 1
    item = target[0]
    item_image = crop_image(image, item['bbox'])
    item_name, item_confidence = recognition(preprocess_image(item_image))
    image_name, image_confidence = recognition(preprocess_image(image))
    if item_name != image_name:
        # print('item name:', item_name, item_confidence, 'image_name:', image_name, image_confidence)
        # landmark_name = item_name if item_confidence > image_confidence else image_name
        if image_confidence > item_confidence:
            landmark_name = image_name
            item['bbox'] = [1, 1, image.size[0]-1, image.size[1]-1]
        else:
            landmark_name = item_name
    else:
        landmark_name = item_name
    landmark_confidence = float(max(item_confidence, image_confidence))
    # print('landmarkname:', landmark_name, 'landmark_conf:', landmark_confidence)
    item['object_finegrained_name'] = landmark_name
    item['landmark'] = {'name': landmark_name, 'confidence': landmark_confidence}
    print('Landmark Recognition Done!')
    return to_serializable(target)
           

           
