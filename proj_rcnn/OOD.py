class ObjectDetectionDataset():
    '''
    A Pytorch Dataset class to load the images and their corresponding annotations.
    
    Returns
    ------------
    images: torch.Tensor of size (B, C, H, W)
    gt bboxes: torch.Tensor of size (B, max_objects, 4)
    gt classes: torch.Tensor of size (B, max_objects)
    '''
    def __init__(self, name2indx, img_dir, truth_dir = None, des_size = (640, 480)):
        self.name2indx = name2indx
        self.truth_dir = truth_dir 
        self.img_dir = img_dir
        self.des_size = des_size
        self.og_width = []
        self.og_height = []
        
        self.img_data_all, self.gt_bboxes_all, self.gt_classes_all = self.get_data()
        
    def __len__(self):
        return self.img_data_all.size(dim=0)
    
    def __getitem__(self, idx):
        return self.img_data_all[idx], self.gt_bboxes_all[idx], self.gt_classes_all[idx]

    def scale_boxes(self, truth_boxes, original_size, target_size):
        target_width, target_height = self.des_size

        wsf = target_width / self.og_width
        hsf = target_height / self.og_height

        scaled_boxes = []
        for box in truth_boxes:
            x_min, y_min, x_max, y_max = box
            scaled_x_min = int(x_min * wsf)
            scaled_y_min = int(y_min * hsf)
            scaled_x_max = int(x_max * wsf)
            scaled_y_max = int(y_max * hsf)
            scaled_boxes.append((scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max))
        return scaled_boxes
    
    
    def tb(self, filename):
        letters = []
        boxes = []
        with open(os.path.join(self.truth_dir, filename), 'r') as file:
            for line in file:
                parts = line.strip().split()
                if parts:
                    if parts[0][0] == '#': 
                        letter = " "
                        last_five_entries = parts[-6:]
                    else:
                        last_five_entries = parts[-5:]
                        letter = last_five_entries[-1][1]
                    letters.append(letter)
                    first_four_entries = [int(entry) for entry in last_five_entries[:4]]
                    tensor_entry = torch.tensor(first_four_entries)
                    boxes.append(tensor_entry)
        return torch.stack(boxes, dim = 0) , letters
           
        
    def get_data(self):
        img_data_all = []
        gt_boxes_all = []
        gt_idxs_all = []
        
        for filename in os.listdir(self.img_dir):
            image = Image.open(os.path.join(self.img_dir, filename))
            self.og_width.append(image.size[0])
            self.og_height.append(image.size[1])
            img_np = np.array(imgage)
            img = resize(img_np, self.des_size)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            img_data_all.append(img_tensor)
        
        for filename in os.listdir(self.truth_dir): 
            if filename.endswith('.txt'): 
                tbs, corresp_classes = self.tb(filename)
                gt_boxes_all.append(tbs)
                gt_idxs_all.append(torch.Tensor([self.name2indx[name] for name in corresp_classes])) 

        # pad bounding boxes and classes so they are of the same size
        gt_bboxes_pad = pad_sequence(gt_boxes_all, batch_first=True, padding_value=-1)
        gt_classes_pad = pad_sequence(gt_idxs_all, batch_first=True, padding_value=-1)
        
        # stack all images
        img_data_stacked = torch.stack(img_data_all, dim=0)
        
        return img_data_stacked.to(dtype=torch.float32), gt_bboxes_pad, gt_classes_pad
