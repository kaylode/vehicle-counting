
class mAP():
    def __init__(self):
        self.reset()
        self.det = {
            'boxes' : [],
            'labels' : [],
            'scores' : []
        }

        self.gt = {
            'boxes' : [],
            'labels' : [],
        }

    def update(self, outputs, targets):
        self.det['boxes'].extend(outputs['det_boxes'])
        self.det['labels'].extend(outputs['det_labels'])
        self.det['scores'].extend(outputs['det_scores'])
        self.gt['boxes'].extend(targets['gt_boxes'])
        self.gt['labels'].extend(targets['gt_labels'])
        self.sample_size += len(outputs['det_boxes'])

    def reset(self):
        self.det = {
            'boxes' : [],
            'labels' : [],
            'scores' : []
        }

        self.gt = {
            'boxes' : [],
            'labels' : [],
        }

        self.sample_size = 0

    def __str__(self):
        return f'mAP: {self.value()}'

    def __len__(self):
        return len(self.sample_size)

    def value(self):
        """
        Calculate the Mean Average Precision (mAP) of detected objects.
        See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation
        :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
        :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
        :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
        :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
        :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
        :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
        :return: list of average precisions for all classes, mean average precision (mAP)
        """
        det_boxes = self.det['boxes']
        det_labels = self.det['labels']
        det_scores = self.det['scores']
        true_boxes = self.gt['boxes']
        true_labels = self.gt['labels']
        true_difficulties = [0 for i in range(len(true_labels))]

        assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
            true_labels) == len(
            true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
        n_classes = len(label_map)

        # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
        true_images = list()
        for i in range(len(true_labels)):
            true_images.extend([i] * true_labels[i].size(0))
        true_images = torch.LongTensor(true_images).to(
            device)  # (n_objects), n_objects is the total no. of objects across all images
        true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
        true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
        true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

        assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

        # Store all detections in a single continuous tensor while keeping track of the image it is from
        det_images = list()
        for i in range(len(det_labels)):
            det_images.extend([i] * det_labels[i].size(0))
        det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
        det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
        det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
        det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

        assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

        # Calculate APs for each class (except background)
        average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
        for c in range(1, n_classes):
            # Extract only objects with this class
            true_class_images = true_images[true_labels == c]  # (n_class_objects)
            true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
            true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
            n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

            # Keep track of which true objects with this class have already been 'detected'
            # So far, none
            true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
                device)  # (n_class_objects)

            # Extract only detections with this class
            det_class_images = det_images[det_labels == c]  # (n_class_detections)
            det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
            det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
            n_class_detections = det_class_boxes.size(0)
            if n_class_detections == 0:
                continue

            # Sort detections in decreasing order of confidence/scores
            det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
            det_class_images = det_class_images[sort_ind]  # (n_class_detections)
            det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

            # In the order of decreasing scores, check if true or false positive
            true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
            false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
            for d in range(n_class_detections):
                this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
                this_image = det_class_images[d]  # (), scalar

                # Find objects in the same image with this class, their difficulties, and whether they have been detected before
                object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
                object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
                # If no such object in this image, then the detection is a false positive
                if object_boxes.size(0) == 0:
                    false_positives[d] = 1
                    continue

                # Find maximum overlap of this detection with objects in this image of this class
                overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
                max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

                # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
                # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
                original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
                # We need 'original_ind' to update 'true_class_boxes_detected'

                # If the maximum overlap is greater than the threshold of 0.5, it's a match
                if max_overlap.item() > 0.5:
                    # If the object it matched with is 'difficult', ignore it
                    if object_difficulties[ind] == 0:
                        # If this object has already not been detected, it's a true positive
                        if true_class_boxes_detected[original_ind] == 0:
                            true_positives[d] = 1
                            true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                        # Otherwise, it's a false positive (since this object is already accounted for)
                        else:
                            false_positives[d] = 1
                # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
                else:
                    false_positives[d] = 1

            # Compute cumulative precision and recall at each detection in the order of decreasing scores
            cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
            cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
            cumul_precision = cumul_true_positives / (
                    cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
            cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

            # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
            recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
            precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
            for i, t in enumerate(recall_thresholds):
                recalls_above_t = cumul_recall >= t
                if recalls_above_t.any():
                    precisions[i] = cumul_precision[recalls_above_t].max()
                else:
                    precisions[i] = 0.
            average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

        # Calculate Mean Average Precision (mAP)
        mean_average_precision = average_precisions.mean().item()

        # Keep class-wise average precisions in a dictionary
        average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

        return {
            'AP': average_precisions,
            'mAP': mean_average_precision}