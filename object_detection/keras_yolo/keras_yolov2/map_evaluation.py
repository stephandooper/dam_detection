from .utils import compute_overlap, compute_ap
import tensorflow as tf
import numpy as np
import tensorflow.keras

# input image dimensions or content is wrong
# pred boxes is wrong

class MapEvaluation(tensorflow.keras.callbacks.Callback):
    """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
    """

    def __init__(self, yolo, generator,
                 labels,
                 iou_threshold=0.5,
                 score_threshold=0.5,
                 save_path=None,
                 period=1,
                 save_best=False,
                 save_name=None,
                 writer=None):

        super().__init__()
        self._yolo = yolo
        self._generator = generator
        self._labels = labels
        self._iou_threshold = iou_threshold
        self._score_threshold = score_threshold
        self._save_path = save_path
        self._period = period
        self._save_best = save_best
        self._save_name = save_name
        self._file_writer = writer

        self.bestMap = 0
        '''
        if not isinstance(self._tensorboard, tensorflow.keras.callbacks.TensorBoard) and self._tensorboard is not None:
            raise ValueError("Tensorboard object must be a instance from keras.callbacks.TensorBoard")
        '''

    def on_epoch_end(self, epoch, logs={}):

        if epoch % self._period == 0 and self._period != 0:
            _map, average_precisions, precision, recall = self.evaluate_map()
            f1 = (2 * precision * recall) / (precision + recall)
            print('\n')
            print("precision, recall, f1:", precision,recall, f1)
            print('\n')
            for label, average_precision in average_precisions.items():
                print(self._yolo.labels[label], '{:.4f}'.format(average_precision))
            print('mAP: {:.4f}'.format(_map))

            if self._save_best and self._save_name is not None and _map > self.bestMap:
                print("mAP improved from {} to {}, saving model to {}.".format(self.bestMap, _map, self._save_name))
                self.bestMap = _map
                self.model.save(self._save_name)
            else:
                print("mAP did not improve from {}.".format(self.bestMap))


            with self._file_writer.as_default():
                tf.summary.scalar('val_mAP', _map, step=epoch)
                tf.summary.scalar('val_precision', precision, step=epoch)
                tf.summary.scalar('val_recall', recall, step=epoch)
                tf.summary.scalar('val_f1', f1, step=epoch)


    def evaluate_map(self):
        average_precisions, precision, recall = self._calc_avg_precisions()
        _map = sum(average_precisions.values()) / len(average_precisions)
		
        return _map, average_precisions, precision, recall


    def _calc_avg_precisions(self):

        # gather all detections and annotations
        all_detections = [[None for _ in range(self._generator.num_classes())]
                          for _ in range(self._generator.size())]
        all_annotations = [[None for _ in range(self._generator.num_classes())]
                           for _ in range(self._generator.size())]

        for i, (raw_image, bbox, class_names) in enumerate(self._generator(mode='map')):
            #print("class maps", class_names)
            raw_image = np.squeeze(raw_image.numpy())
            raw_height, raw_width, _ = raw_image.shape

            # make the boxes and the labels
            pred_boxes = self._yolo.predict(raw_image,
                                            iou_threshold=self._iou_threshold,
                                            score_threshold=self._score_threshold)

            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])

            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin * raw_width, box.ymin * raw_height, box.xmax * raw_width,
                                        box.ymax * raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])

            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(self._generator.num_classes()):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]
            #print("the detections are", all_detections)
            annotations = self.load_annotation(bbox, class_names, self._labels)
            #print("annotations are", annotations)
            
            # copy ground truth to all_annotations
            for label in range(self._generator.num_classes()):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
            #print("generator num classes", self._generator.num_classes())
            
        # compute mAP by comparing all detections and all annotations
        average_precisions = {}

        for label in range(self._generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(self._generator.size()):
                detections = all_detections[i][label]
                
                annotations = all_annotations[i][label]
                num_annotations += annotations.shape[0]
                detected_annotations = []
                
                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]
                    
                    #print("max overlap and iou threshold:, ", max_overlap, self._iou_threshold)
                    #print("assigned annotation and detected_annotation", assigned_annotation, detected_annotations)
                    if max_overlap >= self._iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                print("something went wrong here")
                average_precisions[label] = 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]
            fp = np.sum(false_positives)
            tp = np.sum(true_positives)      

  
            # compute false positives and true positives for ROC
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)
                
            
            # compute recall and precision curve (1 and 0 are added in ap calc in utils)
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
            
            # singular precision/recall metric
            rec = tp / num_annotations
            prec = tp / (tp + fp)
            print("tp and fp:", tp,fp)
            
            # compute average precision
            average_precision = compute_ap(recall, precision)
            average_precisions[label] = average_precision
            #print("average precision computed as: ", average_precisions)
        return average_precisions, prec, rec
    
    # custom load annotaiton function, does not scale ot multiple labels per image
    def load_annotation(self, bbox, class_name, labels):
        bbox = np.squeeze(bbox.numpy())
        # always reshape bbox into a 2d array
        bbox = np.reshape(bbox, (-1,4))
        bbox_dim = bbox.shape
        bbox = np.ravel(bbox, order='F')
        
        class_name = np.squeeze(class_name.numpy(), axis=0)
        class_name = [x.decode("utf-8") for x in class_name]
        class_name = np.array([labels.index(x) for x in class_name]) # convert to integers
    
        
        all_objs = np.reshape( np.concatenate( (bbox, class_name) ), (bbox_dim[0], 5), order='F') 
    
        annots = []
        for obj in all_objs:
            annot = [obj[0], obj[2], obj[1], obj[3], obj[4]]
            annots += [annot]
    
        if len(annots) == 0:
            annots = [[]]
    
        #print("annotations xmin, ymin, xmax, ymax is: ", annots)
        return np.array(annots)
