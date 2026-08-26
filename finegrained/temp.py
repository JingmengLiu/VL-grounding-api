def process_images_inner_i2t(p: StableDiffusionProcessing) -> Processed:    
    # 首先进行了一系列初始化操作
    if isinstance(p.prompt, list):
        assert(len(p.prompt) > 0)
    else:
        assert p.prompt is not None
    raw_image = Image.fromarray(p.init_image)
    p.setup_prompts()
    infotexts = []
    output_text = []
    fine_grained_output = []
    p.init(p.all_prompts)
    prompt = "Briefly describe this image." if p.all_prompts[0] == '' else p.all_prompts[0]
    
    p.iteration = 0
    device = torch.device('cuda')
    state.nextjob()
    p.color_corrections = None
    # TODO!
    image = vis_processors["eval"](raw_image.convert("RGB")).unsqueeze(0).to(device)
    # TODO!
    output_text = lavis_used_model.generate({"image": image, "prompt": prompt}) # 无论是否选择细粒度都先进行image caption
    # output_text = 'Caption is empty now'
    fg_output_dict = {item: [] for item in p.Fine_grained}
    true_face_dict = []
    aircraft_dict = []
    car_dict = []
    if not p.Fine_grained:
        p.enable_fg = False
    if p.enable_fg:
        # 将p.Fine_grained按照首字母排序
        p.Fine_grained.sort()
        det_label_info = p.detect_label_info
        labels = None
        xyxys = None
        obj_none_res = False
        if p.selected_part=='Original image': # 如果选择Original image, 则使用目标检测
            if det_label_info is not None:
                def crop_image(image, xyxys):
                    x1, y1, x2, y2 = xyxys
                    cropped_image = image[y1:y2, x1:x2]
                    return cropped_image
                if 'shape=(' in det_label_info: # 目标检测未检测出结果
                    obj_none_res = True # 这块需要处理一下
                if not obj_none_res:
                    # 将 numpy 数组的字符串表示替换为 Python 列表的字符串表示
                    det_label_info = det_label_info.replace('\n      ', '')
                    det_label_info = det_label_info.replace('dtype=float32)', '')
                    det_label_info = det_label_info.replace('array(', '')
                    
                    # 使用 ast.literal_eval 将字符串转换为列表
                    det_label_info = ast.literal_eval(det_label_info)
                    # 将列表中的子列表转换为 numpy 数组
                    for i, item in enumerate(det_label_info):
                        if isinstance(item, list) and isinstance(item[0], list):
                            det_label_info[i] = np.array(item)
                    labels = det_label_info[0] # 后面可以从labels中提取信息
                    xyxys = det_label_info[1]
                    """# 寻找selected_part在labels中的位置
                    index = labels.index(p.selected_part)
                    xyxys = xyxys[index]
                    # 将xyxys转为整型
                    xyxys = xyxys.astype(int)
                    # 将image按照xyxy裁剪
                    res_image = crop_image(image, xyxys)"""# 后面根据具体的细粒度模型来写逻辑
        # fine_ask_string = f"""Image caption: {output_text[0]}. """ # 逐步增加内容形成最终的prompt
        for choice in p.Fine_grained: #大意是加载好每个模型并存储到对应的全局变量中
            if choice == 'Human face':
                # 先加载
                face_dict_dir = curr_path + "/../weights/face_embeddings_dict.pkl"
                workers = 0 if os.name == 'nt' else 8
                batch_size = 16
                save_path = curr_path + "/../weights/face_embeddings_dict.pkl"
                resnet = InceptionResnetV1(
                        classify=False,
                        pretrained='vggface2'
                    ).to('cpu')
                resnet.eval()
                if not os.path.exists(face_dict_dir):
                    data_dir = curr_path + "/../models/lfw"
                    mtcnn = MTCNN(
                        image_size=160,
                        margin=14,
                        device=device,
                        selection_method='center_weighted_size'
                    )

                    orig_img_ds = datasets.ImageFolder(data_dir, transform=None)
                    orig_img_ds.samples = [
                        (p, p)
                        for p, _ in orig_img_ds.samples
                    ]

                    loader = DataLoader(
                        orig_img_ds,
                        num_workers=workers,
                        batch_size=batch_size,
                        collate_fn=training.collate_pil
                    )
                    crop_paths = []
                    box_probs = []

                    for i, (x, b_paths) in enumerate(loader):
                        crops = [p.replace(data_dir, data_dir + '_cropped') for p in b_paths]
                        mtcnn(x, save_path=crops)
                        crop_paths.extend(crops)
                        print('\r第 {} 批，共 {} 批'.format(i + 1, len(loader)), end='')

                    del mtcnn
                    torch.cuda.empty_cache()
                    trans = transforms.Compose([
                        np.float32,
                        transforms.ToTensor(),
                        fixed_image_standardization
                    ])
                    dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
                    embed_loader = DataLoader(
                        dataset,
                        num_workers=workers,
                        batch_size=batch_size,
                        sampler=SequentialSampler(dataset)
                    )
                    classes = []
                    embeddings = []
                    with torch.no_grad():
                        for xb, yb in embed_loader:
                            xb = xb.to(device)
                            b_embeddings = resnet(xb)
                            b_embeddings = b_embeddings.to('cpu').numpy()
                            classes.extend(yb.numpy())
                            embeddings.extend(b_embeddings)
                    embeddings_dict = dict(zip(crop_paths,embeddings))
                    # 保存embeddings_dict
                    with open(save_path, 'wb') as f:
                        pickle.dump(embeddings_dict, f)
                # 加载embeddings_dict
                with open(save_path, 'rb') as file:
                    embeddings_dict = pickle.load(file)
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
                # 人脸模型加载完毕, 下面根据具体选择情况和目标检测结果进行预测
                resnet.eval()
                mtcnn = MTCNN(
                    image_size=160,
                    margin=14,
                    device=device,
                    selection_method='center_weighted_size'
                )

                # 如果选择了Original image, 则使用目标检测, 对每个检测出'people'的框进行人脸预测
                if p.selected_part=='Original image' and not obj_none_res:
                    person_indices = [i for i, label in enumerate(labels) if 'person' in label and float(label.split(' ')[-1]) > 0.50]
                    for i in person_indices:
                        xyxy = xyxys[i]
                        xyxy = xyxy.astype(int)
                        res_image = crop_image(p.init_image, xyxy)
                        res_image = Image.fromarray(res_image)
                        img = res_image
                        # Get cropped and prewhitened image tensor
                        try:
                            img_cropped = mtcnn(img, save_path='crop.jpg')
                        except RuntimeError:
                            continue
                        if img_cropped is None:
                            continue
                        # Calculate embedding (unsqueeze to add batch dimension)
                        img_embedding = resnet(img_cropped.to('cpu').unsqueeze(0)).detach().numpy()
                        distances = []
                        src_img_paths = []
                        for src_img_path, src_img_embedding in embeddings_dict.items():
                            distances.append(distance(src_img_embedding, img_embedding, distance_metric=0)[0])
                            src_img_paths.append(src_img_path)
                        if np.min(distances) <= 0.6:
                            most_similar_idx = np.argmin(distances)
                            fg_output_dict['Human face'].append(str(i + 1) + ":" + src_img_paths[most_similar_idx].split('/')[-1][:-9].replace("_", " ") + " " + str(round(1-np.min(distances), 2)))
                            true_face_dict.append(src_img_paths[most_similar_idx].split('/')[-1][:-9].replace("_", " "))
                        print('fg_output_dict', fg_output_dict)
                else: # 如果选择了其他part，则直接对其他部位进行预测
                    # img = raw_image
                    img = Image.fromarray(p.detect_image)
                    # Get cropped and prewhitened image tensor
                    img_cropped = mtcnn(img, save_path='crop.jpg')
                    if img_cropped is None:
                        continue
                    # Calculate embedding (unsqueeze to add batch dimension)
                    img_embedding = resnet(img_cropped.to('cpu').unsqueeze(0)).detach().numpy()
                    distances = []
                    src_img_paths = []
                    for src_img_path, src_img_embedding in embeddings_dict.items():
                        distances.append(distance(src_img_embedding, img_embedding, distance_metric=0)[0])
                        # distances.append((torch.tensor(src_img_embedding) - img_embedding).norm().item())
                        src_img_paths.append(src_img_path)
                    if np.min(distances) <= 0.6:
                        most_similar_idx = np.argmin(distances)
                        fg_output_dict['Human face'].append(src_img_paths[most_similar_idx].split('/')[-1][:-9].replace("_", " ") + " " + str(round(1-np.min(distances), 2)))
                        true_face_dict.append(src_img_paths[most_similar_idx].split('/')[-1][:-9].replace("_", " "))
                # if fg_output_dict['Human face'] != []:
                #     fine_ask_string += f"""Please replace the person or people with {'"' + '", "'.join(fg_output_dict['Human face'])+'"'}"""
            elif choice == 'Landmark':
                # def get_file_content_as_base64(raw_image, urlencoded=False):
                #     byte_arr = io.BytesIO()
                #     raw_image.save(byte_arr, format='PNG')
                #     byte_arr = byte_arr.getvalue()
                #     content = base64.b64encode(byte_arr).decode("utf8")
                #     if urlencoded:
                #         content = urllib.parse.quote_plus(content)
                #     return content
                # def get_access_token():
                #     """
                #     使用 AK，SK 生成鉴权签名（Access Token）
                #     :return: access_token，或是None(如果错误)
                #     """
                #     url = "https://aip.baidubce.com/oauth/2.0/token"
                #     params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
                #     return str(requests.post(url, params=params).json().get("access_token"))
                
                # API_KEY = "CGWtGqYGymDpxq8F0xUr6WNl"
                # SECRET_KEY = "1C8kRNEmCi6qvwULhUTbGt2diciHcNlx"
                # translator = Translator()
                # url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/landmark?access_token=" + get_access_token()
                # image_ = get_file_content_as_base64(raw_image, True)
                # payload="image=" + image_
                # headers = {
                #         'Content-Type': 'application/x-www-form-urlencoded',
                #         'Accept': 'application/json'
                #     }
                # response = requests.request("POST", url, headers=headers, data=payload)
                # output_landmark = json.loads(response.text)
                # response.close()
                # if 'result' in output_landmark and 'landmark' in output_landmark['result'] and output_landmark['result']['landmark'] != '':
                #     translation = translator.translate(output_landmark['result']['landmark'], dest='en')
                #     fg_output_dict['Landmark'].append(translation.text)
                # else:
                landmark_output = lavis_used_model.generate({"image": image, "prompt": "Where is this place?"})
                # landmark_output = lavis_used_model.generate({"image": image, "prompt": "What is the landmark in this image?"})
                print('landmark_output', landmark_output)
                start_time = time.time()
                tokenized_query = tokenizer.tokenize(landmark_output[0])
                fg_output_dict['Landmark'].append(bm25.get_top_n(tokenized_query, corpus, n=1)[0])
                module1_time = time.time() - start_time
                print(f"OVEN time: {module1_time} seconds")
            elif choice == 'Aircraft':
                # airplane
                aircraft_net.eval()
                img = Image.fromarray(p.detect_image)
                if p.selected_part=='Original image' and not obj_none_res:
                    aircraft_indices = [i for i, label in enumerate(labels) if 'aircraft' in label and float(label.split(' ')[-1]) > 0.50]
                    for i in aircraft_indices:
                        xyxy = xyxys[i]
                        xyxy = xyxy.astype(int)
                        res_image = crop_image(p.init_image, xyxy)
                        res_image = Image.fromarray(res_image)
                        aircraft_max_value, aircraft_nname = CMAL_inference(res_image, os.path.join(curr_path, '..', 'repositories', 'CMAL', 'annot.json'), aircraft_net)
                        if aircraft_max_value > 0.98:
                            aircraft_dict.append(aircraft_nname)
                            fg_output_dict['Aircraft'].append(str(i + 1) + ":" + aircraft_dict[-1])
                            print('fg_output_dict', fg_output_dict)
                else:
                    aircraft_max_value, aircraft_nname = CMAL_inference(img, os.path.join(curr_path, '..', 'repositories', 'CMAL', 'annot.json'), aircraft_net)
                    if aircraft_max_value > 0.98:
                        aircraft_dict.append(aircraft_nname)
                        fg_output_dict['Aircraft'].append(aircraft_dict[-1])
                        print('fg_output_dict', fg_output_dict)
            elif choice == 'Car':
                # airplane
                img = Image.fromarray(p.detect_image)
                if p.selected_part=='Original image' and not obj_none_res:
                    aircraft_indices = [i for i, label in enumerate(labels) if 'car' in label and float(label.split(' ')[-1]) > 0.50]
                    for i in aircraft_indices:
                        xyxy = xyxys[i]
                        xyxy = xyxy.astype(int)
                        res_image = crop_image(p.init_image, xyxy)
                        res_image = Image.fromarray(res_image)
                        car_dict.append(car_net(car_args, car_model, car_state, res_image))
                        fg_output_dict['Car'].append(str(i + 1) + ":" + car_dict[-1])
                        print('fg_output_dict', fg_output_dict)
                else:
                    car_dict.append(car_net(car_args, car_model, car_state, img))
                    fg_output_dict['Car'].append(car_dict[-1])
                    print('fg_output_dict', fg_output_dict)
            else:
                pass
        # 细粒度Caption输出
        # gpt version
        '''
        client = OpenAI()
        max_retries = 10
        wait_time = 1
        print("GPT-START")
        start_time = time.time()
        prompt = """
        Giving you an image caption and external information about the image, please merge the external information into the caption.
        Please replace related words in the caption with external information and provide a new caption. Keep the rest of the sentence unchanged as much as possible.
        Image caption: {}
        External information: {}
        Only output the new caption.
        New image caption:
        """
        caption  = output_text[0]
        # human_name = ', '.join(fg_output_dict['Human face']) if 'Human face' in fg_output_dict and fg_output_dict["Human face"] != [] else None
        human_name = ', '.join(true_face_dict) if true_face_dict != [] else None
        landmark_name = fg_output_dict["Landmark"][0] if 'Landmark' in fg_output_dict and fg_output_dict["Landmark"] != [] else None
        aircraft_name = ', '.join(aircraft_dict) if aircraft_dict != [] else None
        car_name = ', '.join(car_dict) if car_dict != [] else None
        external_info = ''
        if human_name:
            external_info += '\nPerson name: {}'.format(human_name)

        if landmark_name:
            external_info += '\nLandmark: {}'.format(landmark_name)

        if aircraft_name:
            external_info += '\nAircraft: {}'.format(aircraft_name)
        if car_name:
            external_info += '\nCar: {}'.format(car_name)
        retries = 0
        while retries < max_retries:
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                    {"role": "user", "content" : prompt.format(caption, external_info)}
                    ],
                    timeout=20
                )
                fine_grained_output = response.choices[0].message.content.strip('"')
                
                print(prompt.format(caption, external_info))
                print("Output:", fine_grained_output)
                break

            except Exception as e:
                print(f"Error: {e}")
                retries += 1
                print(f"Retry {retries} times")
                time.sleep(wait_time)
        module2_time = time.time() - start_time
        print(f"GPT time: {module2_time} seconds")
        '''
        
        # fine_grained_output = lavis_used_model.generate({"image": image_used, "prompt": f"""{fine_ask_string}"""})

        # llama-2 api version
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://10.208.40.54:7861/v1/",
            # base_url="http://llama2:7861/v1/",
        )
        max_retries = 10
        wait_time = 1
        print("Llama-START")
        start_time = time.time()
        prompt = """Giving you an image caption and external information about the image, please merge the external information into the caption.
Please replace related words in the caption with external information and provide a new caption. Keep the rest of the sentence unchanged as much as possible.
Image caption: {}
External information: {}
Only output the new caption.
New image caption:
        """
        content1 = '''Giving you a sentence and external information about it, please merge the external information into the sentence.
Please REPLACE  corresponding words in the sentence with every each external information, and keep the rest of the sentence unchanged as much as possible.
Do NOT contain the type of external information. 
Please only output as "OUTPUT: [the merged sentence]".

For example:
Sentence: an asian walking with an umbrella on the beach in a sunny day
External information:
Person name: Zhang Ziyi
Landmark: Golden Beach
Output: Zhang Ziyi walking with an umbrella on Golden Beach in a sunny day.
        '''

        content2 = '''Please only output as "OUTPUT: [the merged sentence]".

Sentence: {}
External information: {}
        '''

        caption  = output_text[0]
        # human_name = ', '.join(fg_output_dict['Human face']) if 'Human face' in fg_output_dict and fg_output_dict["Human face"] != [] else None
        human_name = ', '.join(true_face_dict) if true_face_dict != [] else None
        landmark_name = fg_output_dict["Landmark"][0] if 'Landmark' in fg_output_dict and fg_output_dict["Landmark"] != [] else None
        aircraft_name = ', '.join(aircraft_dict) if aircraft_dict != [] else None
        car_name = ', '.join(car_dict) if car_dict != [] else None
        external_info = ''
        if human_name:
            external_info += '\nPerson name: {}'.format(human_name)

        if landmark_name:
            external_info += '\nLandmark: {}'.format(landmark_name)

        if aircraft_name:
            external_info += '\nAircraft: {}'.format(aircraft_name)
        if car_name:
            external_info += '\nCar: {}'.format(car_name)
        retries = 0
        while retries < max_retries:
            try:
                # response = client.chat.completions.create(
                #     model="gpt-3.5-turbo-1106",
                #     messages=[
                #     {"role": "user", "content" : prompt.format(caption, external_info)}
                #     ],
                #     timeout=20
                # )'
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {
                            "role": "user",
                            "content": content1,
                        },
                        {
                            "role": "assistant",
                            "content": "Sure, I'd be happy to help! What's the sentence and external information you'd like me to merge?",
                        },
                        {
                            "role": "user",
                            "content": content2.format(caption, external_info),
                        }
                    ],
                    model="gpt-3.5-turbo",
                )
                fine_grained_output = response.choices[0].message.content.strip('"')
                # 如果fine_grained_output以"OUTPUT: "开头，则去掉
                if fine_grained_output.startswith("OUTPUT: "):
                    fine_grained_output = fine_grained_output[8:]
                print(prompt.format(caption, external_info))
                print("Output:", fine_grained_output)
                break

            except Exception as e:
                print(f"Error: {e}")
                retries += 1
                print(f"Retry {retries} times")
                time.sleep(wait_time)
        module2_time = time.time() - start_time
        print(f"Llama time: {module2_time} seconds")

    # tokenized_query = tokenizer.tokenize(fg_output) if fg_output is not None else tokenizer.tokenize(output_text[0])
    # fg_output = bm25.get_top_n(tokenized_query, corpus, n=1)[0]
    text_list2_ = ''
    text_list3_ = ''
    print('fine_grained_output:', fine_grained_output)
    if p.enable_fg and p.selected_part:
        text_list2_ = fg_output_dict
        if not all(not value for value in fg_output_dict.values()):
            text_list3_ = fine_grained_output
        else:
            text_list3_ = output_text[0]
    else:
        text_list2_ = ''
        text_list3_ = ''
    
    res = Processed(
        p,
        text_list=output_text[0] if output_text else '', # General caption框输出结果
        text_list2 = text_list2_, # Fine-grained label框输出结果
        text_list3 = text_list3_, # Fine-grained caption框输出结果
        # ↑逻辑改成dist全部为[]
        info=infotexts[0] if infotexts else '',
        infotexts=infotexts,
    )

    return res
