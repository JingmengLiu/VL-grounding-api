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
from concurrent.futures import ThreadPoolExecutor
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
 

def _normalize_entity_name(value):
    value = re.sub(r'[^a-z0-9]+', ' ', str(value or '').lower())
    return ' '.join(value.split())


def _canonicalize_flag_logo_name(value, symbol_type=None):
    """Remove visual-symbol wrappers while preserving the entity's display name."""
    name = str(value or '').strip().strip('"\'`').strip()
    name = re.sub(r'[\s\u00a0]+', ' ', name)
    name = re.sub(r'[\s.,;:!?]+$', '', name).strip()
    if _normalize_entity_name(name) in {'', 'unknown', 'none', 'null'}:
        return 'unknown'

    symbol_phrases = (
        r'coat\s+of\s+arms|party\s+symbol|organization\s+symbol|'
        r'national\s+flag|official\s+flag|official\s+logo|'
        r'flag|logo|emblem|seal|symbol'
    )
    prefix_patterns = (
        rf'^(?:the\s+)?(?:{symbol_phrases})\s+of\s+(?:the\s+)?(.+)$',
        rf'^(?:the\s+)?(?:{symbol_phrases})\s+for\s+(?:the\s+)?(.+)$',
    )
    for pattern in prefix_patterns:
        match = re.match(pattern, name, flags=re.I)
        if match:
            return match.group(1).strip()

    suffix_match = re.match(
        rf'^(.+?)\s+(?:{symbol_phrases})$', name, flags=re.I
    )
    if suffix_match:
        return suffix_match.group(1).strip()

    return name


def _parse_flag_logo_vlm_result(raw_text):
    """Parse the structured VLM answer; tolerate fenced JSON and plain text."""
    text = (raw_text or '').strip()
    if not text:
        return {
            'symbol_type': 'unknown', 'raw_name': '', 'name': 'unknown', 'confidence': 0.0,
            'specificity': 'unknown', 'matched_candidate': None,
        }

    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, flags=re.I | re.S)
    json_text = fenced.group(1) if fenced else text
    if not fenced:
        object_match = re.search(r'\{.*\}', text, flags=re.S)
        if object_match:
            json_text = object_match.group(0)

    try:
        value = json.loads(json_text)
    except (TypeError, ValueError):
        # Compatibility with a caption service that still returns one name.
        return {
            'symbol_type': 'unknown', 'raw_name': text,
            'name': _canonicalize_flag_logo_name(text), 'confidence': 0.5,
            'specificity': 'unknown', 'matched_candidate': None,
            'parse_fallback': True,
        }

    if not isinstance(value, dict):
        value = {}
    try:
        confidence = min(1.0, max(0.0, float(value.get('confidence', 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    specificity = str(value.get('specificity') or 'unknown').strip().lower()
    if specificity not in {'specific', 'generic', 'unknown'}:
        specificity = 'unknown'
    symbol_type = str(value.get('symbol_type') or 'unknown').strip().lower()
    raw_name = str(value.get('name') or 'unknown').strip()
    return {
        'symbol_type': symbol_type,
        'raw_name': raw_name,
        'name': _canonicalize_flag_logo_name(raw_name, symbol_type),
        'confidence': confidence,
        'specificity': specificity,
        'matched_candidate': value.get('matched_candidate'),
    }


def _build_flag_logo_vlm_prompt(mode, candidates):
    candidate_lines = '\n'.join(
        f"{index}. {candidate['class_name']}"
        for index, candidate in enumerate(candidates, start=1)
    ) or '（无候选信息，请独立判断）'
    requested_kind = '旗帜' if mode == 'flag' else '标志、徽章或印章'
    return f"""请识别裁剪图像中的{requested_kind}。候选结果如下：
{candidate_lines}

请独立观察图像，不要因为候选中存在某个名称就强行选择。若图像只是国家国徽、国旗、通用政府徽章或印章，不得推断为某个具体政府机构。无法可靠判断时返回 unknown。未提供候选时，matched_candidate 必须为 null。

name 只能填写实体本身的简短英文名称，不得包含 flag of、logo of、coat of arms of、emblem of、seal of、symbol of、official logo 等描述性前后缀。例如美国旗帜填写 United States，俄罗斯国徽填写 Russian，Indian National Congress 的标志填写 Indian National Congress。符号类别只通过 symbol_type 表达。

只输出一个 JSON 对象，不要输出解释或 Markdown：
{{
  "symbol_type": "flag|logo|coat_of_arms|seal|party_symbol|organization_symbol|unknown",
  "name": "英文名称或unknown",
  "confidence": 0.0,
  "specificity": "specific|generic|unknown",
  "matched_candidate": "与候选完全对应时填写候选英文名称，否则为null"
}}"""


def _find_vlm_candidate(vlm_result, candidates):
    requested = vlm_result.get('matched_candidate') or vlm_result.get('name')
    normalized = _normalize_entity_name(
        _canonicalize_flag_logo_name(requested, vlm_result.get('symbol_type'))
    )
    if not normalized:
        return None
    return next(
        (candidate for candidate in candidates
         if _normalize_entity_name(
             _canonicalize_flag_logo_name(
                 candidate.get('class_name'), vlm_result.get('symbol_type')
             )
         ) == normalized),
        None,
    )


def _fuse_flag_logo_results(
    prediction,
    vlm_result,
    siglip_strong_threshold,
    siglip_weak_threshold,
    vlm_strong_threshold,
    vlm_weak_threshold,
):
    """Fuse incomparable model scores with conservative, explainable rules."""
    candidates = prediction.get('top_k') or [prediction]
    top1 = candidates[0]
    similarity = float(top1['similarity'])
    vlm_confidence = float(vlm_result.get('confidence', 0.0))
    vlm_name = str(vlm_result.get('name') or 'unknown').strip()
    vlm_unknown = _normalize_entity_name(vlm_name) in {'', 'unknown', 'none', 'null'}
    matched_candidate = _find_vlm_candidate(vlm_result, candidates)
    generic_symbol = (
        vlm_result.get('specificity') == 'generic'
        or vlm_result.get('symbol_type') in {'coat_of_arms', 'seal'}
    )

    if matched_candidate and (
        float(matched_candidate['similarity']) >= siglip_weak_threshold
        or vlm_confidence >= vlm_weak_threshold
    ):
        return {
            'accepted': True,
            'decision': 'model_agreement',
            'source': 'siglip_vlm_agreement',
            'name': matched_candidate['class_name'],
            'candidate': matched_candidate,
            'confidence_level': 'high',
        }

    if generic_symbol and not vlm_unknown and vlm_confidence >= vlm_strong_threshold \
            and similarity < siglip_strong_threshold:
        return {
            'accepted': True,
            'decision': 'vlm_generic_override',
            'source': 'vlm',
            'name': vlm_name,
            'candidate': None,
            'confidence_level': 'high',
        }

    siglip_strong = similarity >= siglip_strong_threshold
    vlm_strong = vlm_confidence >= vlm_strong_threshold and not vlm_unknown

    if siglip_strong and not vlm_strong:
        return {
            'accepted': True,
            'decision': 'siglip_strong',
            'source': 'siglip2_two_mode',
            'name': top1['class_name'],
            'candidate': top1,
            'confidence_level': 'high',
        }

    if vlm_strong and similarity < siglip_strong_threshold:
        return {
            'accepted': True,
            'decision': 'vlm_strong',
            'source': 'vlm',
            'name': vlm_name,
            'candidate': None,
            'confidence_level': 'high',
        }

    if siglip_strong and vlm_strong:
        return {
            'accepted': True,
            'decision': 'both_high_vlm_selected',
            'source': 'vlm',
            'name': vlm_name,
            'candidate': None,
            'confidence_level': 'high',
        }

    return {
        'accepted': True,
        'decision': 'both_low_siglip_selected',
        'source': 'siglip2_two_mode',
        'name': top1['class_name'],
        'candidate': top1,
        'confidence_level': 'low',
    }


def flag_logo_recognition(
    recognizer,
    mode,
    image,
    target,
    siglip_strong_threshold=0.75,
    caption_api_url=None,
    siglip_weak_threshold=0.60,
    vlm_strong_threshold=0.80,
    vlm_weak_threshold=0.60,
    top_k=3,
):
    """Recognize flag/logo crops with parallel SigLIP2 and VLM inference."""
    if mode not in ('flag', 'logo'):
        raise ValueError(f'Unsupported flag/logo mode: {mode}')

    print(f'Start recognizing {mode.capitalize()} (SigLIP2 + VLM fusion).')
    if len(target) == 0:
        return to_serializable(target)

    source_metadata = {
        'flags': ('Flag', 'country-flags-in-the-wild'),
        'Logo': ('Logo', 'Logo-2K+'),
        'Party': ('Party', 'Party'),
        'Organization': ('Organization', 'Organization'),
    }

    for item in target:
        item_image = crop_image(image, item['bbox'])
        prompt = _build_flag_logo_vlm_prompt(mode, [])
        vlm_result = {
            'symbol_type': 'unknown', 'raw_name': '', 'name': 'unknown', 'confidence': 0.0,
            'specificity': 'unknown', 'matched_candidate': None,
            'available': False,
        }
        # Run the independent visual judgements concurrently. The VLM does not
        # see the retrieval result, which avoids candidate anchoring.
        with ThreadPoolExecutor(max_workers=2) as executor:
            siglip_future = executor.submit(recognizer.predict, item_image, mode, top_k)
            vlm_future = None
            if caption_api_url:
                vlm_future = executor.submit(
                    call_caption_api, item_image, prompt, caption_api_url
                )
            prediction = siglip_future.result()
            if vlm_future:
                try:
                    raw_vlm_result = vlm_future.result()
                    vlm_result = {**_parse_flag_logo_vlm_result(raw_vlm_result), 'available': True}
                except Exception as e:
                    print('caption-api failed:', e)
                    vlm_result['error'] = str(e)

        candidates = prediction.get('top_k') or [prediction]

        decision = _fuse_flag_logo_results(
            prediction,
            vlm_result,
            float(siglip_strong_threshold),
            float(siglip_weak_threshold),
            float(vlm_strong_threshold),
            float(vlm_weak_threshold),
        )

        top1 = candidates[0]
        result_info = {
            'name': decision['name'],
            'confidence': float(top1['similarity']),
            'similarity': float(top1['similarity']),
            'score_type': 'cosine_similarity',
            'siglip2_class_key': top1['class_key'],
            'siglip2_source_category': top1['source_category'],
            'siglip2_top_k': candidates,
            'vlm': vlm_result,
            'recognition_source': decision['source'],
            'decision': decision['decision'],
            'accepted': decision['accepted'],
            'final_confidence_level': decision['confidence_level'],
        }
        if top1.get('qid'):
            result_info['siglip2_qid'] = top1['qid']
        item[mode] = result_info
        item['flag_logo_fusion'] = result_info

        if decision['accepted']:
            item['object_finegrained_name'] = decision['name']
            candidate = decision.get('candidate')
            source_category = candidate.get('source_category') if candidate else None
            entity_metadata = source_metadata.get(source_category)
            if entity_metadata:
                entity_type, dataset = entity_metadata
                item['object_finegrained_type'] = entity_type
                item['object_finegrained_dataset'] = dataset
                # All packaged bank IDs match the corresponding Source_ID:
                # numeric Logo/Flag IDs and Wikidata QIDs for Party/Organization.
                item['object_finegrained_id'] = str(candidate['id'])

    print(f'{mode.capitalize()} Recognition Done!')
    return to_serializable(target)


def logo_recognition(recognizer, image, target, *fusion_args):
    return flag_logo_recognition(
        recognizer, 'logo', image, target, *fusion_args
    )


def flag_recognition(recognizer, image, target, *fusion_args):
    return flag_logo_recognition(
        recognizer, 'flag', image, target, *fusion_args
    )
        

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
           

           
