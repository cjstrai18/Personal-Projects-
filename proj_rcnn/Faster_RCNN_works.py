import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as F
import torch.nn.functional as F
from torchvision.models.detection import anchor_utils
from torchvision.models import resnet50, ResNet50_Weights
from skimage.transform import resize
from skimage import io
from torchvision.ops import box_iou
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import matplotlib.pyplot as plt
import string
import OOD 
import os
from torchvision import ops
import matplotlib.patches as patches
class ObjectDetectionDataset():
    '''
    A Pytorch Dataset class to load the images and their corresponding annotations.
    
    Returns
    ------------
    images: torch.Tensor of size (B, C, H, W)
    gt bboxes: torch.Tensor of size (B, max_objects, 4)
    gt classes: torch.Tensor of size (B, max_objects)
    '''
    def __init__(self, name2indx, img_dir, truth_dir = None, des_size = (480,640)):
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
            img_np = np.array(image)
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

class ProposalModule(nn.Module):
    def __init__(self, in_features, hidden_dim=512, n_anchors=9, p_dropout=0.3):
        super().__init__()
        self.n_anchors = n_anchors
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p_dropout)
        self.conf_head = nn.Conv2d(hidden_dim, n_anchors, kernel_size=1)
        self.reg_head = nn.Conv2d(hidden_dim, n_anchors * 4, kernel_size=1)
        
    def forward(self, feature_map, pos_anc_ind=None, neg_anc_ind=None, pos_anc_coords=None):
        # determine mode
        if pos_anc_ind is None or neg_anc_ind is None or pos_anc_coords is None:
            mode = 'eval'
        else:
            mode = 'train'
            
        out = self.conv1(feature_map)
        out = F.relu(self.dropout(out))
        
        reg_offsets_pred = self.reg_head(out) # (B, A*4, hmap, wmap)
        conf_scores_pred = self.conf_head(out) # (B, A, hmap, wmap)
        
        if mode == 'train': 
            # get conf scores 
            conf_scores_pos = conf_scores_pred.flatten()[pos_anc_ind]
            conf_scores_neg = conf_scores_pred.flatten()[neg_anc_ind]
            # get offsets for +ve anchors
            offsets_pos = reg_offsets_pred.contiguous().view(-1, 4)[pos_anc_ind]
            # generate proposals using offsets
            proposals = self.generate_proposals(pos_anc_coords, offsets_pos)
            
            return conf_scores_pos, conf_scores_neg, offsets_pos, proposals
            
        elif mode == 'eval':
            #print('prop mode is eval')
            return conf_scores_pred, reg_offsets_pred
        
    def generate_proposals(self, anc, offsets):  
        anc = ops.box_convert(anc, in_fmt='xyxy', out_fmt='cxcywh') 

        proposals_ = torch.zeros_like(anc)
        proposals_[:,0] = anc[:,0] + offsets[:,0]*anc[:,2]
        proposals_[:,1] = anc[:,1] + offsets[:,1]*anc[:,3]
        proposals_[:,2] = anc[:,2] * torch.exp(offsets[:,2])
        proposals_[:,3] = anc[:,3] * torch.exp(offsets[:,3])

        proposals = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')

        return proposals
        


#########################################################################################################################################################################
# ########################################################################################################################################################################

    

class RegionProposalNetwork(nn.Module):
    def __init__(self, img_size, out_size, out_channels):
        super().__init__()
        
        self.img_height, self.img_width = img_size
        self.out_h, self.out_w = out_size
        
        # downsampling scale factor 
        self.width_scale_factor = self.img_width // self.out_w
        self.height_scale_factor = self.img_height // self.out_h 
        
        # scales and ratios for anchor boxes
        self.anc_scales = [1, 2, 3]
        self.anc_ratios = [0.25, .5, 1]
        self.n_anc_boxes = len(self.anc_scales) * len(self.anc_ratios)
        
        # IoU thresholds for +ve and -ve anchors
        self.pos_thresh = 0.7
        self.neg_thresh = 0.3
        
        # weights for loss
        self.w_conf = 1
        self.w_reg = 5
        
        self.feature_extractor = FeatureExtractor()
        self.proposal_module = ProposalModule(out_channels, n_anchors=self.n_anc_boxes)
        
    def forward(self, images, gt_bboxes, gt_classes):
        batch_size = images.size(dim=0)
        feature_map = self.feature_extractor(images)
        
        # generate anchors
        anc_pts_x, anc_pts_y = self.gen_anc_centers(out_size=(self.out_h, self.out_w))
        anc_base = self.gen_anc_base(anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
        anc_boxes_all = anc_base.repeat(batch_size, 1, 1, 1, 1)
        
        # get positive and negative anchors amongst other things
        gt_bboxes_proj = self.project_bboxes(gt_bboxes, self.width_scale_factor, self.height_scale_factor, mode='p2a')
        
        positive_anc_ind, negative_anc_ind, GT_conf_scores, \
        GT_offsets, GT_class_pos, positive_anc_coords, \
        negative_anc_coords, positive_anc_ind_sep = self.get_req_anchors(anc_boxes_all, gt_bboxes_proj, gt_classes)
        
        # pass through the proposal module
        conf_scores_pos, conf_scores_neg, offsets_pos, proposals = self.proposal_module(feature_map, positive_anc_ind, \
                                                                                        negative_anc_ind, positive_anc_coords)
        
        cls_loss = self.calc_cls_loss(conf_scores_pos, conf_scores_neg, GT_conf_scores, batch_size)
        reg_loss = self.calc_bbox_reg_loss(GT_offsets, offsets_pos, batch_size)
        
        total_rpn_loss = self.w_conf * cls_loss + self.w_reg * reg_loss
        
        return total_rpn_loss, feature_map, proposals, positive_anc_ind_sep, GT_class_pos
    
    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):
        with torch.no_grad():
            batch_size = images.size(dim=0)
            feature_map = self.feature_extractor(images)

            # generate anchors
            anc_pts_x, anc_pts_y = self.gen_anc_centers(out_size=(self.out_h, self.out_w))
            anc_base = self.gen_anc_base(anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
            anc_boxes_all = anc_base.repeat(batch_size, 1, 1, 1, 1)
            anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)
            #print('INfRPN ab:', anc_boxes_all)

            # get conf scores and offsets
            conf_scores_pred, offsets_pred = self.proposal_module(feature_map)
            conf_scores_pred = conf_scores_pred.reshape(batch_size, -1)
            offsets_pred = offsets_pred.reshape(batch_size, -1, 4)

            # filter out proposals based on conf threshold and nms threshold for each image
            proposals_final = []
            conf_scores_final = []
            for i in range(batch_size):
                conf_scores = torch.sigmoid(conf_scores_pred[i])
                offsets = offsets_pred[i]
                anc_boxes = anc_boxes_flat[i]
                proposals = self.proposal_module.generate_proposals(anc_boxes, offsets)
                #print("RPN props:", proposals)
                # filter based on confidence threshold
                conf_idx = torch.where(conf_scores >= conf_thresh)[0]
                conf_scores_pos = conf_scores[conf_idx]
                proposals_pos = proposals[conf_idx]
                # filter based on nms threshold
                nms_idx = ops.nms(proposals_pos, conf_scores_pos, nms_thresh)
                #print('RPN nms_idx:',nms_idx )
                conf_scores_pos = conf_scores_pos[nms_idx]
                proposals_pos = proposals_pos[nms_idx]

                proposals_final.append(proposals_pos)
                conf_scores_final.append(conf_scores_pos)
            
        return proposals_final, conf_scores_final, feature_map
    
    def gen_anc_centers(self,out_size):
        out_h, out_w = out_size

        anc_pts_x = torch.arange(0, out_w) + 0.5
        anc_pts_y = torch.arange(0, out_h) + 0.5

        return anc_pts_x, anc_pts_y
    
    def gen_anc_base(self, anc_pts_x, anc_pts_y, anc_scales, anc_ratios, out_size):
        n_anc_boxes = len(anc_scales) * len(anc_ratios)
        anc_base = torch.zeros(1, anc_pts_x.size(dim=0) \
                                  , anc_pts_y.size(dim=0), n_anc_boxes, 4) # shape - [1, Hmap, Wmap, n_anchor_boxes, 4]

        for ix, xc in enumerate(anc_pts_x):
            for jx, yc in enumerate(anc_pts_y):
                anc_boxes = torch.zeros((n_anc_boxes, 4))
                c = 0
                for i, scale in enumerate(anc_scales):
                    for j, ratio in enumerate(anc_ratios):
                        w = scale * ratio
                        h = scale

                        xmin = xc - w / 2
                        ymin = yc - h / 2
                        xmax = xc + w / 2
                        ymax = yc + h / 2

                        anc_boxes[c, :] = torch.Tensor([xmin, ymin, xmax, ymax])
                        c += 1

                anc_base[:, ix, jx, :] = ops.clip_boxes_to_image(anc_boxes, size=out_size)

        return anc_base
    
    def project_bboxes(self, bboxes, width_scale_factor, height_scale_factor, mode='a2p'):
        assert mode in ['a2p', 'p2a']

        batch_size = bboxes.size(dim=0)
        proj_bboxes = bboxes.clone().reshape(batch_size, -1, 4).to(torch.float)
        invalid_bbox_mask = (proj_bboxes == -1) # indicating padded bboxes

        if mode == 'a2p':
            # activation map to pixel image
            proj_bboxes[:, :, [0, 2]] *= width_scale_factor
            proj_bboxes[:, :, [1, 3]] *= height_scale_factor
        else:
            # pixel image to activation map
            proj_bboxes[:, :, [0, 2]] /= width_scale_factor
            proj_bboxes[:, :, [1, 3]] /= height_scale_factor

        proj_bboxes.masked_fill_(invalid_bbox_mask, -1) # fill padded bboxes back with -1
        proj_bboxes.resize_as_(bboxes)

        return proj_bboxes
    
    def get_req_anchors(self,anc_boxes_all, gt_bboxes_all, gt_classes_all, pos_thresh=0.7, neg_thresh=0.2):
        '''
        Prepare necessary data required for training

        Input
        ------
        anc_boxes_all - torch.Tensor of shape (B, w_amap, h_amap, n_anchor_boxes, 4)
            all anchor boxes for a batch of images
        gt_bboxes_all - torch.Tensor of shape (B, max_objects, 4)
            padded ground truth boxes for a batch of images
        gt_classes_all - torch.Tensor of shape (B, max_objects)
            padded ground truth classes for a batch of images

        Returns
        ---------
        positive_anc_ind -  torch.Tensor of shape (n_pos,)
            flattened positive indices for all the images in the batch
        negative_anc_ind - torch.Tensor of shape (n_pos,)
            flattened positive indices for all the images in the batch
        GT_conf_scores - torch.Tensor of shape (n_pos,), IoU scores of +ve anchors
        GT_offsets -  torch.Tensor of shape (n_pos, 4),
            offsets between +ve anchors and their corresponding ground truth boxes
        GT_class_pos - torch.Tensor of shape (n_pos,)
            mapped classes of +ve anchors
        positive_anc_coords - (n_pos, 4) coords of +ve anchors (for visualization)
        negative_anc_coords - (n_pos, 4) coords of -ve anchors (for visualization)
        positive_anc_ind_sep - list of indices to keep track of +ve anchors
        '''
        # get the size and shape parameters
        B, w_amap, h_amap, A, _ = anc_boxes_all.shape
        N = gt_bboxes_all.shape[1] # max number of groundtruth bboxes in a batch

        # get total number of anchor boxes in a single image
        tot_anc_boxes = A * w_amap * h_amap

        # get the iou matrix which contains iou of every anchor box
        # against all the groundtruth bboxes in an image
        iou_mat = self.get_iou_mat(B, anc_boxes_all, gt_bboxes_all)

        # for every groundtruth bbox in an image, find the iou 
        # with the anchor box which it overlaps the most
        max_iou_per_gt_box, _ = iou_mat.max(dim=1, keepdim=True)

        # get positive anchor boxes

        # condition 1: the anchor box with the max iou for every gt bbox
        positive_anc_mask = torch.logical_and(iou_mat == max_iou_per_gt_box, max_iou_per_gt_box > 0) 
        # condition 2: anchor boxes with iou above a threshold with any of the gt bboxes
        positive_anc_mask = torch.logical_or(positive_anc_mask, iou_mat > pos_thresh)

        positive_anc_ind_sep = torch.where(positive_anc_mask)[0] # get separate indices in the batch
        # combine all the batches and get the idxs of the +ve anchor boxes
        positive_anc_mask = positive_anc_mask.flatten(start_dim=0, end_dim=1)
        positive_anc_ind = torch.where(positive_anc_mask)[0]

        # for every anchor box, get the iou and the idx of the
        # gt bbox it overlaps with the most
        max_iou_per_anc, max_iou_per_anc_ind = iou_mat.max(dim=-1)
        max_iou_per_anc = max_iou_per_anc.flatten(start_dim=0, end_dim=1)

        # get iou scores of the +ve anchor boxes
        GT_conf_scores = max_iou_per_anc[positive_anc_ind]

        # get gt classes of the +ve anchor boxes

        # expand gt classes to map against every anchor box
        gt_classes_expand = gt_classes_all.view(B, 1, N).expand(B, tot_anc_boxes, N)
        # for every anchor box, consider only the class of the gt bbox it overlaps with the most
        GT_class = torch.gather(gt_classes_expand, -1, max_iou_per_anc_ind.unsqueeze(-1)).squeeze(-1)
        # combine all the batches and get the mapped classes of the +ve anchor boxes
        GT_class = GT_class.flatten(start_dim=0, end_dim=1)
        GT_class_pos = GT_class[positive_anc_ind]

        # get gt bbox coordinates of the +ve anchor boxes

        # expand all the gt bboxes to map against every anchor box
        gt_bboxes_expand = gt_bboxes_all.view(B, 1, N, 4).expand(B, tot_anc_boxes, N, 4)
        # for every anchor box, consider only the coordinates of the gt bbox it overlaps with the most
        GT_bboxes = torch.gather(gt_bboxes_expand, -2, max_iou_per_anc_ind.reshape(B, tot_anc_boxes, 1, 1).repeat(1, 1, 1, 4))
        # combine all the batches and get the mapped gt bbox coordinates of the +ve anchor boxes
        GT_bboxes = GT_bboxes.flatten(start_dim=0, end_dim=2)
        GT_bboxes_pos = GT_bboxes[positive_anc_ind]

        # get coordinates of +ve anc boxes
        anc_boxes_flat = anc_boxes_all.flatten(start_dim=0, end_dim=-2) # flatten all the anchor boxes
        positive_anc_coords = anc_boxes_flat[positive_anc_ind]

        # calculate gt offsets
        GT_offsets = self.calc_gt_offsets(positive_anc_coords, GT_bboxes_pos)

        # get -ve anchors

        # condition: select the anchor boxes with max iou less than the threshold
        negative_anc_mask = (max_iou_per_anc < neg_thresh)
        negative_anc_ind = torch.where(negative_anc_mask)[0]
        # sample -ve samples to match the +ve samples
        negative_anc_ind = negative_anc_ind[torch.randint(0, negative_anc_ind.shape[0], (positive_anc_ind.shape[0],))]
        negative_anc_coords = anc_boxes_flat[negative_anc_ind]

        return positive_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class_pos, \
             positive_anc_coords, negative_anc_coords, positive_anc_ind_sep
    
    def get_iou_mat(self, batch_size, anc_boxes_all, gt_bboxes_all):
    
        # flatten anchor boxes
        anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)
        # get total anchor boxes for a single image
        tot_anc_boxes = anc_boxes_flat.size(dim=1)

        # create a placeholder to compute IoUs amongst the boxes
        ious_mat = torch.zeros((batch_size, tot_anc_boxes, gt_bboxes_all.size(dim=1)))

        # compute IoU of the anc boxes with the gt boxes for all the images
        for i in range(batch_size):
            gt_bboxes = gt_bboxes_all[i]
            anc_boxes = anc_boxes_flat[i]
            ious_mat[i, :] = ops.box_iou(anc_boxes, gt_bboxes)

        return ious_mat
    
    def calc_gt_offsets(self, pos_anc_coords, gt_bbox_mapping):
        pos_anc_coords = ops.box_convert(pos_anc_coords, in_fmt='xyxy', out_fmt='cxcywh')
        gt_bbox_mapping = ops.box_convert(gt_bbox_mapping, in_fmt='xyxy', out_fmt='cxcywh')

        gt_cx, gt_cy, gt_w, gt_h = gt_bbox_mapping[:, 0], gt_bbox_mapping[:, 1], gt_bbox_mapping[:, 2], gt_bbox_mapping[:, 3]
        anc_cx, anc_cy, anc_w, anc_h = pos_anc_coords[:, 0], pos_anc_coords[:, 1], pos_anc_coords[:, 2], pos_anc_coords[:, 3]

        tx_ = (gt_cx - anc_cx)/anc_w
        ty_ = (gt_cy - anc_cy)/anc_h
        tw_ = torch.log(gt_w / anc_w)
        th_ = torch.log(gt_h / anc_h)

        return torch.stack([tx_, ty_, tw_, th_], dim=-1)
    
    def calc_cls_loss(self, conf_scores_pos, conf_scores_neg, GT_conf, batch_size):
        # Assuming `conf_scores_pos` and `conf_scores_neg` are torch tensors
        pos_loss = F.binary_cross_entropy_with_logits(conf_scores_pos, GT_conf)
        neg_loss = F.binary_cross_entropy_with_logits(conf_scores_neg, GT_conf)
        
        # Compute mean classification loss over batch
        cls_loss = (pos_loss + neg_loss) / batch_size
        
        return cls_loss
    
    def calc_bbox_reg_loss(self, GT_offsets, offsets_pos, batch_size):
        # Calculate Mean Squared Error (MSE) loss between predicted and ground truth offsets
        criterion = nn.MSELoss()

        # Compute the loss between predicted offsets (offsets_pos) and ground truth offsets (GT_offsets)
        reg_loss = criterion(offsets_pos, GT_offsets)

        # Compute mean loss over the batch
        reg_loss = reg_loss / batch_size

        return reg_loss
    
    

########################################################################################################################
# #######################################################################################################################
class FeatureExtractor(nn.Module): 
    def __init__(self, custom_layers = None):
        super().__init__()
        self.custom_layers = custom_layers 
        # Load pre-trained ResNet50 model
        self.model = torchvision.models.resnet50(pretrained=True)
        #custom_layers is expected to be a list of ints, specifying the indices of the layers the backbone will consist of
        #default layers used are those following each of the 4 blocks in the resnet50 model
        
        
    def forward(self, images, display = False):
        model = self.model

        if self.custom_layers is not None:
            bb_layers = [getattr(model, f'layer{i}') for i in self.custom_layers]
        else:
            bb_layers = list(model.children())[:8]

        backbone = nn.Sequential(*bb_layers)
        backbone.eval()
        im = images.to(torch.float32)

        with torch.no_grad():
            feat = backbone(im)

        if display:
            self.display_map(feat)
        return feat
    


#########################################################################################################################################################################
# ########################################################################################################################################################################  

    
class ClassificationModule(nn.Module):
    def __init__(self, out_channels, n_classes, roi_size, hidden_dim=512, p_dropout=0.3):
        super().__init__()        
        self.roi_size = roi_size
        # hidden network
        self.avg_pool = nn.AvgPool2d(self.roi_size)
        self.fc = nn.Linear(out_channels, hidden_dim)
        self.dropout = nn.Dropout(p_dropout)
        
        # define classification head
        self.cls_head = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, feature_map, proposals_list, gt_classes=None):
        
        if gt_classes is None:
            mode = 'eval'
        else:
            mode = 'train'
        
        # apply roi pooling on proposals followed by avg pooling
        roi_out = ops.roi_pool(feature_map, proposals_list, self.roi_size)
        roi_out = self.avg_pool(roi_out)
        
        # flatten the output
        roi_out = roi_out.squeeze(-1).squeeze(-1)
        
        # pass the output through the hidden network
        out = self.fc(roi_out)
        out = F.relu(self.dropout(out))
        
        # get the classification scores
        cls_scores = self.cls_head(out)
        
        if mode == 'eval':
            return cls_scores
        
        # compute cross entropy loss
        cls_loss = F.cross_entropy(cls_scores, gt_classes.long())
        
        return cls_loss
#########################################################################################################################################################################
# ########################################################################################################################################################################

    
class TwoStageDetector(nn.Module):
    def __init__(self, img_size, out_size, out_channels, n_classes, roi_size):
        super().__init__() 
        self.rpn = RegionProposalNetwork(img_size, out_size, out_channels)
        self.classifier = ClassificationModule(out_channels, n_classes, roi_size)
        
    def forward(self, images, gt_bboxes, gt_classes):
        total_rpn_loss, feature_map, proposals, \
        positive_anc_ind_sep, GT_class_pos = self.rpn(images, gt_bboxes, gt_classes)
        
        # get separate proposals for each sample
        pos_proposals_list = []
        batch_size = images.size(dim=0)
        for idx in range(batch_size):
            proposal_idxs = torch.where(positive_anc_ind_sep == idx)[0]
            proposals_sep = proposals[proposal_idxs].detach().clone()
            pos_proposals_list.append(proposals_sep)
        
        cls_loss = self.classifier(feature_map, pos_proposals_list, GT_class_pos)
        total_loss = cls_loss + total_rpn_loss
        
        return total_loss
    
    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):
        print('inf_TSD')
        batch_size = images.size(dim=0)
        proposals_final, conf_scores_final, feature_map = self.rpn.inference(images, conf_thresh, nms_thresh)
        cls_scores = self.classifier(feature_map, proposals_final)
        #print('inf_TSD2')
        # convert scores into probability
        cls_probs = F.softmax(cls_scores, dim=-1)
        # get classes with highest probability
        classes_all = torch.argmax(cls_probs, dim=-1)
        #print('hey',classes_all)
        classes_final = []
        # slice classes to map to their corresponding image
        c = 0
        for i in range(batch_size):
            n_proposals = len(proposals_final[i]) # get the number of proposals for each image
            #print('num final:',n_proposals)
            classes_final.append(classes_all[c: c+n_proposals])
            c += n_proposals
            
        return proposals_final, conf_scores_final, classes_final
