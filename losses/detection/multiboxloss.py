from torch import nn
from utils.utils import change_box_order, find_jaccard_overlap
from torch.autograd import Variable
from ..focalloss import FocalLoss
import torch


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, device = None, **kwargs):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy

        # Default boxes of the model
        self.priors_xy = change_box_order(priors_cxcy,order='cxcy2xyxy')
        
        self.set_attribute(kwargs)
        self.set_loss_func()
        self.device = torch.device("cuda" if device is not None else "cpu")

    def set_attribute(self, kwargs):
        self.use_focal_loss = False
        self.alpha = 1. # total loss = conf_loss + alpha*loc_loss
        self.threshold = 0.5 # background label threshold
        for i,j in kwargs.items():
            setattr(self, i, j)

    def set_loss_func(self):
        self.loc_loss_func = nn.SmoothL1Loss()
        if self.use_focal_loss:
            # Focal loss
            self.conf_loss_func = FocalLoss(gamma=2)
        else:
            # Cross Entropy loss + Hard Negative Mining 
            self.conf_loss_func = nn.CrossEntropyLoss(reduction='none')         
            self.neg_pos_ratio = 3  # neg = neg_pos_ratio * pos
            


    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors. Format (x,y,w,h)
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)  # (N, 8732)
        ignored_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device) # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            # Change format of bounding boxes
            boxes[i] = change_box_order(boxes[i], order = 'xywh2xyxy')
            
            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy, order='xyxy')  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(self.device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Ignore between 0.4 and 0.5 which the model is confused, set to -1
            ignore = (overlap_for_each_prior > 0.4) & (overlap_for_each_prior < 0.5)  # ignore ious between [0.4,0.5]
            ignored_classes[i][ignore] = 1

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(change_box_order(boxes[i][object_for_each_prior],order ='xyxy2cxcy'), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive and not ignored (object/non-background)
        positive_priors = (true_classes > 0) & (ignored_classes != 1)  # (N, 8732)

        # Only ignored priors
        ignored_priors = ignored_classes == 1 # (N, 8732)
        
        #Number of positive prior per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        
        # ===================================
        # =         LOCALIZATION LOSS       =
        # ===================================
        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.loc_loss_func(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        
        
        # ===================================
        # =         CONFIDENCE LOSS         =
        # ===================================
        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Focal Loss
        if self.use_focal_loss:
            pos_neg_priors = ignored_classes != 1
            predicted_scores = predicted_scores[pos_neg_priors]
            true_classes = true_classes[pos_neg_priors]
            conf_loss = self.conf_loss_func(predicted_scores.view(-1, n_classes), true_classes.view(-1))
        else:
            # Hard-negative mining examples
            # Number of  hard-negative priors per image
        
            n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

            # First, find the loss for all priors
            conf_loss_all = self.conf_loss_func(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
            conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

            # We already know which priors are positive
            conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

            # Next, find which priors are hard-negative
            # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
            conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
            conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
            conf_loss_neg[ignored_priors] = 0 # ignored priors are ignored
            conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
            hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(self.device)  # (N, 8732)
            hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
            conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

            # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
            conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar
        
        
        # Loss = 0 when there is no positive match in the image
        # which cause inf loss

        #print(" C: {} || B: {} ".format(conf_loss.item(), loc_loss.item()))
        # TOTAL LOSS
        total_loss = conf_loss + self.alpha * loc_loss
        return total_loss, {'T': total_loss.item(), 'C': conf_loss.item(), 'B': loc_loss.item()}


# Mapping coordinate to priors to regress
def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2])*1.0 / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(abs(cxcy[:, 2:] / priors_cxcy[:, 2:])) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] *1.0 / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h