import tensorflow as tf

from object_detection.core import box_list, box_list_ops
from object_detection.core import standard_fields as fields
import sys
sys.path.append("/notebooks/Transcription/tf-crnn")
from tf_crnn.model import deep_bidirectional_lstm, get_words_from_chars
from tf_crnn.config import  CONST
from tf_crnn.data_handler import padding_inputs_width
from functools import partial
from utils import shape_utils

import numpy as np # debug
sys.path.append("/notebooks/text-renderer/generation/")
import data_util as du
from pprint import pprint

class CRNNFlags:
    """ CRNN flags used to add or modify functionalities of CRNN."""

    def __init__(self,
        replace_detections_with_groundtruth=False,
        train_on_detections_and_groundtruth=False,
        compute_only_per_example_metrics=False,
        explicitely_recompute_field_features=False,
        dump_cropped_fields_to_image_file=False,
        metrics_verbose=False,
        dump_metrics_input_to_tfrecord=False,
        dump_metrics_input_to_tfrecord_using_groundtruth=False):
        """ Object initialization.
        Args:
            replace_detections_with_groundtruth: Train the CRNN using the groundtruth boxes instead of the
                detection boxes. The purpose of this operation is rule out the error propagating from
                the detection model in the training of CRNN.
                Cannot set it on together with train_on_detections_and_groundtruth.
            train_on_detections_and_groundtruth: Train the CRNN using both the groundtruth boxes and
                the detection boxes. The purpose of this operation is to augment the data observed by CRNN
                at every step, thus speeding up training. Cannot set it on together with
                replace_detections_with_groundtruth.
            compute_only_per_example_metrics: Set this flag to print to console metrics after each form
                is processed (i.e. a CRNN batch). Warning: enabling this operation resets the metric value
                after each form is seen, meaning that at the end of the whole evaluation all metrics will hold
                their initial value.
            explicitely_recompute_field_features: CRNN utilizes tf.image.crop_and_resize internally on the last feature
                extractor feature map in order to extract features for each field in a form.
                This has shown to work well in practice. However, with this flag it's possible to explicitely
                recompute field features by applying tf.image.crop_and_resize on the original input image and
                running again the feature extractor. Warning: this might require extra memory.
            dump_cropped_fields_to_image_file: This flag only works in conjunction with explicitely_recompute_field_features.
                You can use this flag to debug what a detected field looks like on the original image before it is passed
                through the feature extractor.
            metrics_verbose: Enabling this flag will print to the console the arguments used to compute the metrics for every
                field (usually couples of predictions and groundtruth).
            dump_metrics_input_to_tfrecord: Dump to a tfrecord every detected field that is used to compute metrics.
                Note: in contrast to dump_cropped_fields_to_image_file, only detected fields that are actually used during
                metric computation are dumped. This is useful for comparing with other architectures.
            dump_metrics_input_to_tfrecord_using_groundtruth: Dump to a tfrecord every groundtruth field (image) that is used
                to compute metrics. This is useful for comparing with other architectures.

        """
        if replace_detections_with_groundtruth and train_on_detections_and_groundtruth:
            raise ValueError("""Invalid values for CRNN flags: cannot set replace_detections_with_groundtruth
                and train_on_detections_and_groundtruth both to TRUE""")
        self.replace_detections_with_groundtruth = replace_detections_with_groundtruth
        self.train_on_detections_and_groundtruth = train_on_detections_and_groundtruth
        self.compute_only_per_example_metrics = compute_only_per_example_metrics
        self.explicitely_recompute_field_features = explicitely_recompute_field_features
        self.dump_cropped_fields_to_image_file = dump_cropped_fields_to_image_file
        self.metrics_verbose = metrics_verbose
        self.dump_metrics_input_to_tfrecord = dump_metrics_input_to_tfrecord
        self.dump_metrics_input_to_tfrecord_using_groundtruth = dump_metrics_input_to_tfrecord_using_groundtruth

class CRNN:
    """ Implements the CRNN graph for transcription. The input is the last layer of the feature extactor and
    the detection boxes coming from stage 2.

    The first operation consists of extracting field features from the last feature extractor map using the detection boxes.
    The function that operates this is crop_feature_map(). All field features are resized and padded to match the same size,
    so that they can be stacked up in a batch. The final size is self._crop_size.

    The next operation is feeding the cropped field features to a bidirectional lstm where the sequence axis is the 'width'
    dimension of the image.

    Note: this implementation does NOT support more than one input image (i.e. the image batch size is 1). Due to the
    high resolution needed for transcription, one single image with Resnet101 as feature extractor is already quite
    memory-demanding. In contrast, even though the there is only one input image, CRNN extracts several crops from
    its feature map, meaning that the preferred way to increase batch size is to increase the image's anumber of field objects
    to transcribe.


    TODO: move this dual evaluation in another class.

    Details on two-streams evaluation implementation:

    In order to evaluate the SYNTH and REAL datasets, we need to separate the two from the
    input pipeline (we can only have one eval spec, so the input pipeline has to combine the two
    datasets into one). In order to do this, the input pipeline keeps track of whether the read
    image is SYNTH or REAL. More precisely, we use a different notation and call them source and
    target respectively, since no specific concept of neither 'synth' nor 'real' is used and
    any coupled dataset can be given as sorce and target (the terminology comes from domain
    adaptation). The variable that keeps the stream type is self.input_features['is_source_metrics'].

    The Tensorflow estimator executes the update op of every metric for every example and calls
    the variable read at the end of the input stream. A naive approach would be to condition both
    operations on input_features['is_source_metrics']. However, because of this conditioning, the variable read could
    only return one value (depending on the condition), since it is requested only once.
    With this approach, it is not possible to retrieve both metric values. Note that
    the variable read does need to be the one under the condition because tensorflow forbids any other access
    to prevent bugs.

    My solution mimics the implementation of coco metrics: the update op merely stores the parameters
    needed for each example evaluation. We maintain two datastructures, one for source and one for target,
    which are filled according to input_features['is_source_metrics'].
    At variable read, we can finally evaluate metrics and read them without any conditioning.
    A complication is due to the fact that metrics are written in tensorflow and not numpy,
    as required by this approach.
    To work around this, metric evaluation occurs by creating two sessions, one for each stream.

    Variables:
        self._metric_names: A list used to establish an arbitrary ordering of the metric keys.
            We need the ordering to decide which one is the first metric to evaluate.
            All the other metrics will depend on the first one. The first metric carries out the
            evaluation of ALL the metrics and stores the result in self._metrics.
        self._metrics: A dictionary of numpy metrics:
            1) 'eval/precision',
            2) 'eval/recall',
            3) 'eval/CER',
            4) 'eval/precision/synth',
            5) 'eval/recall/synth',
            6) 'eval/CER/synth',
    """
    def _init_tables(self, parameters):
        """ Get the alphabet hash tables. This function is called when initializing a session
        for either train or eval streams.

        Args:
            parameters: A Params object, a dictionary with the following keys MUST be provided:
                1) alphabet,
                2) alphabet_decoding
        Returns:
            A tuple of two HashTables, the first one being the encoder and the second one the decoder.

        """
        keys = [c for c in parameters.alphabet.encode('latin1')]
        values = parameters.alphabet_codes
        table_str2int = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.int64,
                    value_dtype=tf.int64), 0)#-1) TODO: The default value should not be 0

        keys = tf.cast(parameters.alphabet_decoding_codes, tf.int64)
        values = [c for c in parameters.alphabet_decoding]
        table_int2str = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(keys, values), '?')
        return table_str2int, table_int2str


    def __init__(self, parameters, detection_model, target_assigner, template_assigner,
        crop_size, start_at_step, backprop_feature_map, backprop_detection, flags=None):
        """Object Initialization.

        Args:
            parameters: A Params object, it should be constructed with a dictionary containing the
                following keys:
                    1) alphabet,
                    2) alphabet_decoding,
                    3) num_corpora,
                    4) keep_prob,
                    5) nb_logprob,
                    6) top_paths
                Please refer to the Params documentation for information on their meaning.
            detection_model: The detection model. The only supported model is FasterRCNNMetaArch.
            target_assigner: The assigner used to map detections to groundtruth. In EVAL mode,
                this is also used to map groundtruth to detections.
            template_assigner: The assigner used to map detections (or possibly groundtruth) to
                the template spaces.
            crop_size: The shape of the field features extracted from the last feature extractor
                feature map. Please note that the features height on the feature map are always resized
                to crop_size[0], while the features width is not distorted in any way, with the exception
                that the extracted crops have double the number of feature cells along the width, in order
                to maximize the probability that there is a valid path for the ctc_loss. Only afterwards this
                extracted crop is padded with zeros to crop_size[1]. You
                can refer to crop_feature_map(). In case the resized resized crops are longer than
                crop_size[1], padding is not possible and hence the features width is clipped to
                crop_size[1].
            start_at_step: The global step value at which the CRNN should start training. The CRNN is
                as if it were not there until then. During EVAL, this parameter is ignored and the
                CRNN is always enabled.
            backprop_feature_map: whether the gradient should flow from the CRNN to the feature
                extractor.
            backprop_detection: whether the gradient should flow from CRNN to the detection model.
                the function connecting CRNN to the detection model is tf.image.crop_and_resize.
                Therefore, the gradient computation is defined by that function.
            flags: a CRNNFLags object containing flags that partially modify the behavior of CRNN.
        """

        self._crop_size = [int(d) for d in crop_size]
        self._start_at_step = start_at_step
        self._parameters = parameters
        self._detection_model = detection_model
        self._backprop_feature_map = backprop_feature_map
        self._backprop_detection = backprop_detection
        self._target_assigner = target_assigner
        self._template_assigner = template_assigner

        if not flags:
            flags = CRNNFLags()
        self.flags = flags

        # The custom alphabet encoder and decoder
        self._table_str2int, self._table_int2str = self._init_tables(parameters)

        # Return zero loss in predict and eval mode
        self._zero_loss = tf.constant(0, dtype=tf.float32)

        # This null symbol is used to pad predictions or groundtruth to a specific size for
        # metric evaluation (precision and recall). Note that this symbol is not in the alphabet,
        # so it will encoded as table_str2int's default value
        self.NULL = '?'


        # Placeholder for eval ops, mainly used to match tf.cond branches
        self.no_eval_op = {
            'eval/precision' : (tf.constant(0, dtype=tf.float32), tf.no_op()),
            'eval/recall' : (tf.constant(0, dtype=tf.float32), tf.no_op()),
            'eval/CER' : (tf.constant(0, dtype=tf.float32), tf.no_op()),
            'eval/precision/synth' : (tf.constant(0, dtype=tf.float32), tf.no_op()),
            'eval/recall/synth' : (tf.constant(0, dtype=tf.float32), tf.no_op()),
            'eval/CER/synth' : (tf.constant(0, dtype=tf.float32), tf.no_op())
        }

        # Arbitrary ordering of the metrics. The ordering is used to determine which is the first metric to be evaluated in the dual-stream eval.
        self._metric_names = [
            'eval/precision',
            'eval/recall',
            'eval/CER',
            'eval/precision/synth',
            'eval/recall/synth',
            'eval/CER/synth',
        ]

        # Being the third stage of the architecture, CRNN takes care of postprocessing stage two.
        # However, if CRNN is disabled, there's no need to postprocess detections.
        self.no_postprocessing = {
            fields.DetectionResultFields.detection_boxes : tf.constant(0, dtype=tf.float32),
            fields.DetectionResultFields.detection_scores : tf.constant(0, dtype=tf.float32),
            fields.DetectionResultFields.detection_corpora : tf.constant(0, dtype=tf.int32),
            fields.DetectionResultFields.num_detections : tf.constant(0, dtype=tf.float32)
        }

        # Metric Args Lists: We store predictions instead of computing metrics directly for
        # 2-streams evaluation. Indeed there is no way of computing multiple metric sets in one eval
        # pass, because the resulting metrics would be unfetchable.
        self._source_predictions, self._target_predictions = [], []


    def no_result_fn(self, detections_dict):
        """Build a placeholder prediction of CRNN. It can be used to return a null prediction.

        Args:
            detections_dict: the postprocessed detections to be integrated in the placeholder result
        Returns:
            A placeholder comforming to a CRNN prediction as far as type goes and with the
            given detections in it. Structure: [loss, transcription_dict, eval_metric_ops]
        """
        return lambda : self.no_forward_pass(detections_dict) + [self.no_eval_op]

    def no_forward_pass(self, detections_dict):
        """Build a placeholder forward pass of CRNN.

        Args:
            detections_dict: the postprocessed detections to be integrated in the placeholder result
        Returns:
            A placeholder comforming to a CRNN forward pass as far as type goes and with the
            given detections in it.
        """
        transcriptions_dict = {
            'raw_predictions': tf.constant(0, dtype=tf.int64),
            'labels': tf.constant('', dtype=tf.string),
            'seq_len_inputs': 0,
            'prob': tf.constant([[0], [0]], dtype=tf.float32),
            fields.TranscriptionResultFields.score: tf.constant([[0]], dtype=tf.float32),
            fields.TranscriptionResultFields.words: tf.constant([['']], dtype=tf.string)
        }
        transcriptions_dict.update(detections_dict)
        return [self._zero_loss, transcriptions_dict]


    def predict(self, prediction_dict, true_image_shapes, mode):
        """Build the CRNN computational graph.

        Args:
            prediction_dict: the detections coming from stage 2.
            true_image_shapes: the shape of the input image.
            mode: train, eval or predict.
        Returns:
            A list of [loss, transcription_dict, eval_metric_ops], where any or all of those
            values could be a placeholder depending on the mode and run-time conditions.

            As far as the latter go, there are two checks being performed: one is that
            the global step is bigger than the given self._start_at_step; the other
            ensures that there is at least one detection coming from stage 2 that
            matches a groundtruth object. No lstm layer is run if any of those checks
            fail.

            Concerning the mode instead:
                1) on TRAIN: only 'loss' is valid.
                2) on EVAL: only 'transcription_dict' and 'eval_metric_ops' are valid.
                3) on PREDICT: only 'transcription_dict' is valid.
        """
        # Catch root variable scope in order to access variables external to CRNN.
        self._debug_root_variable_scope = tf.get_variable_scope()
        with tf.variable_scope('crnn'):
            predict_fn = partial(self._predict, mode=mode, prediction_dict=prediction_dict,
                true_image_shapes=true_image_shapes)
            # CRNN's Condition 1: Global Step condition
            if mode == tf.estimator.ModeKeys.TRAIN:
                global_step = tf.train.get_or_create_global_step()
                disabled = tf.less(global_step, self._start_at_step)
            else:
                # In EVAL mode CRNN is always enabled. This ensures that stage 2 is always post-processed.
                # The placeholder condition is used to preserve namespaces in EVAL for loading variables.
                disabled = tf.constant(False, dtype=tf.bool)
            return tf.cond(disabled, self.no_result_fn(self.no_postprocessing),
                predict_fn, name="StepCond")

    def _predict(self, prediction_dict, true_image_shapes, mode):
        detection_model = self._detection_model
        # Postprocess and unpad detections.
        detections_dict = detection_model._postprocess_box_classifier(
            prediction_dict['refined_box_encodings'],
            prediction_dict['class_predictions_with_background'],
            prediction_dict['proposal_boxes'],
            prediction_dict['num_proposals'],
            true_image_shapes)
        num_detections = tf.cast(detections_dict[
            fields.DetectionResultFields.num_detections][0], tf.int32)
        normalized_detection_boxes = detections_dict[
          fields.DetectionResultFields.detection_boxes][0][:num_detections]
        rpn_features_to_crop = prediction_dict['rpn_features_to_crop']
        detection_scores = detections_dict[
            fields.DetectionResultFields.detection_scores][0][:num_detections]
        # Placeholder detection corpora.
        detections_dict[fields.DetectionResultFields.detection_corpora] = tf.constant([[0]], dtype=tf.int32)
        padded_matched_transcriptions = tf.constant('', dtype=tf.string)
        # Remove detection classes since text detection is not a multiclass problem
        detections_dict.pop(fields.DetectionResultFields.detection_classes)

        normalized_boxlist = box_list.BoxList(normalized_detection_boxes)

        # Fetch groundtruth. We maintain both normalized and absolute groundtruth
        # according to the use.
        if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN]:
            gt_boxlists, gt_classes, _, gt_weights, gt_transcriptions = detection_model._format_groundtruth_data(true_image_shapes,
                stage='transcription')
            normalized_gt_boxlist = box_list.BoxList(tf.placeholder(shape=(1, 4), dtype=tf.float32))
            normalize_gt_fn = lambda: box_list_ops.to_normalized_coordinates(gt_boxlists[0],
                true_image_shapes[0, 0], true_image_shapes[0, 1]).get()
            # Guard for examples with no objects to detect (box_list_ops throws an exception)
            normalized_gt_boxlist.set(tf.cond(gt_boxlists[0].num_boxes() > 0, normalize_gt_fn,
                lambda: gt_boxlists[0].get()))

            if mode == tf.estimator.ModeKeys.TRAIN:
                if self.flags.replace_detections_with_groundtruth:
                    normalized_boxlist = normalized_gt_boxlist
                    num_detections = normalized_gt_boxlist.num_boxes()
                if self.flags.train_on_detections_and_groundtruth:
                    normalized_boxlist = box_list_ops.concatenate([normalized_boxlist, normalized_gt_boxlist])
                    num_detections = num_detections + normalized_gt_boxlist.num_boxes()


        # Template Assignment (boxes with IOU bigger than 0.05 to some template space are mapped to that space)
        # TODO: investigate if assignments on normal coordinates are allowed, since this is not the way it was
        # done on FasterRCNN.
        template_boxlist = box_list.BoxList(detection_model.current_template_boxes)
        (_, _, _, _, match) = self._template_assigner.assign(normalized_boxlist, template_boxlist)
        template_corpora = detection_model.current_template_corpora
        ## A tensor of shape [num_detections]. Each padded_detection_corpora[i] is the corpus type of the detection_i.
        ## If the detection has no type (i.e. probably a false positive detection), then padded_detection_corpora[i]
        ## is -1.
        padded_detection_corpora = match.gather_based_on_match(template_corpora, -1, -1)

        # The name of CRNN's Condition 2. This condition ensures that the loss is computed on a non-empty batch.
        BATCH_COND = 'BatchCond'

        if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN]:
            detection_boxlist = box_list_ops.to_absolute_coordinates(normalized_boxlist,
                true_image_shapes[0, 0], true_image_shapes[0, 1])

            detection_boxlist.add_field(fields.BoxListFields.scores, detection_scores)
            detection_boxlist.add_field(fields.BoxListFields.corpus, padded_detection_corpora)

            (_, cls_weights, _, _, match) = self._target_assigner.assign(detection_boxlist,
                gt_boxlists[0], groundtruth_weights=gt_weights[0])

            # A tensor of shape [num_detections]. Each padded_matched_transcriptions[i] holds the transcription target
            # string of the detection_i. In case detection_i has no transcription target, self.NULL string is assigned instead.
            padded_matched_transcriptions = match.gather_based_on_match(gt_transcriptions[0], self.NULL, self.NULL)

            detection_boxlist.add_field(fields.BoxListFields.groundtruth_transcription, padded_matched_transcriptions)

            positive_indicator = match.matched_column_indicator()
            valid_indicator = cls_weights > 0
            # TODO: rewrite this step so that it's transparent
            sampled_indices = detection_model._second_stage_sampler.subsample(
                valid_indicator,
                None,
                positive_indicator,
                stage="transcription")

            def train_forward_pass():
                sampled_boxlist = box_list_ops.boolean_mask(detection_boxlist, sampled_indices)

                # Replace detections with matched detections
                normalized_sampled_boxlist = box_list_ops.to_normalized_coordinates(sampled_boxlist,
                    true_image_shapes[0, 0], true_image_shapes[0, 1])
                sampled_padded_boxlist = normalized_sampled_boxlist
                normalized_detection_boxes = sampled_padded_boxlist.get()
                matched_transcriptions = sampled_padded_boxlist.get_field(fields.BoxListFields.groundtruth_transcription)
                detection_scores = sampled_padded_boxlist.get_field(fields.BoxListFields.scores)
                detection_corpora = sampled_padded_boxlist.get_field(fields.BoxListFields.corpus)
                num_detections = sampled_boxlist.num_boxes()
                sparse_code_target = self.str2code(matched_transcriptions)

                transcriptions_dict = self._predict_lstm(rpn_features_to_crop, normalized_detection_boxes, matched_transcriptions,
                        detection_scores, detection_corpora, num_detections, mode)

                return [self.loss(transcriptions_dict, sparse_code_target), transcriptions_dict]

            def eval_forward_pass():
                transcriptions_dict = self._predict_lstm(rpn_features_to_crop, normalized_detection_boxes,
                    padded_matched_transcriptions, detection_scores, padded_detection_corpora, num_detections, mode)
                return [self.zero_loss, transcriptions_dict]

            if mode == tf.estimator.ModeKeys.TRAIN:
                # Check that at least one detection matches groundtruth
                loss, predictions_dict = tf.cond(tf.equal(tf.shape(sampled_indices)[0], 0),
                    lambda : self.no_forward_pass(detections_dict), train_forward_pass, name=BATCH_COND)
                eval_metric_ops = self.no_eval_op
            else:
                # Check that there is at least one detection (there might be none due to nms)
                loss, predictions_dict = tf.cond(tf.equal(tf.shape(normalized_detection_boxes)[0], 0), #tf.constant(False, dtype=tf.bool),
                    lambda : self.no_forward_pass(detections_dict), eval_forward_pass, name=BATCH_COND)

                source_update_op = lambda *args: self._source_predictions.append(args)
                target_update_op = lambda *args: self._target_predictions.append(args)
                common_args = [
                    predictions_dict['words'],
                    predictions_dict['detection_boxes'],
                    padded_matched_transcriptions,
                    sampled_indices,
                    normalized_gt_boxlist.get(),
                    gt_transcriptions[0],
                    padded_detection_corpora]
                update_op = tf.cond(self.input_features['is_source_metrics'],
                    lambda: tf.py_func(source_update_op, common_args, []),
                    lambda: tf.py_func(target_update_op, common_args, []))

                # This var does the actual metric evaluation and stores the result in self._metrics
                first_var = tf.py_func(self._first_value_op, [], tf.float32)
                eval_metric_ops = {self._metric_names[0]: (first_var, update_op)}

                with tf.control_dependencies([first_var]):
                    for metric in self._metric_names[1:]:
                        eval_metric_ops[metric] = (tf.py_func(lambda m: self._metrics[m.decode('latin1')], [metric], tf.float32),
                            update_op)

            return [loss, predictions_dict, eval_metric_ops]


        predict_fn = lambda : [self.zero_loss, self._predict_lstm(rpn_features_to_crop, normalized_detection_boxes,
                padded_matched_transcriptions, detection_scores, padded_detection_corpora, num_detections, mode),
                self.no_eval_op]
        return tf.cond(tf.constant(True, dtype=tf.bool), predict_fn,
            self.no_result_fn(detections_dict), name=BATCH_COND)

    def crop_feature_map(self, features_to_crop, bboxes):
      output_height, output_width = self._crop_size

      def _keep_aspect_ratio_crop_and_resize(args):
        bbox, crop_width = args
        fixed_height_crop = tf.image.crop_and_resize(features_to_crop,
          tf.expand_dims(bbox, axis=0), [0], [output_height, crop_width])
        padded_crop = tf.pad(fixed_height_crop[0],
          [[0, 0], [0, output_width - crop_width], [0, 0]], "CONSTANT")
        return padded_crop

      num_feat_map_cells = tf.cast(tf.shape(features_to_crop)[2], dtype=tf.float32) * (bboxes[:, 3] - bboxes[:, 1])
      crop_widths = tf.math.minimum(tf.cast(tf.round(2.0 * num_feat_map_cells), tf.int32),
        output_width)
      crop_widths = tf.math.maximum(crop_widths, 1)

      return shape_utils.static_or_dynamic_map_fn(
              _keep_aspect_ratio_crop_and_resize,
              elems=[bboxes, crop_widths],
              dtype=tf.float32,
              parallel_iterations=self._detection_model._parallel_iterations), crop_widths

    def crop_feature_map_debug(self, img, bboxes, crop_size):
      output_height, output_width = crop_size

      def _keep_aspect_ratio_crop_and_resize(args):
        bbox, crop_width = args
        fixed_height_crop = tf.image.crop_and_resize(img,
          tf.expand_dims(bbox, axis=0), [0], [output_height, crop_width])
        padded_crop = tf.pad(fixed_height_crop[0],
          [[0, 0], [0, output_width - crop_width], [0, 0]], "CONSTANT")
        return padded_crop

      aspect_ratios = (bboxes[:, 3] - bboxes[:, 1]) / (bboxes[:, 2] - bboxes[:, 0])
      crop_widths = tf.math.minimum(tf.cast(tf.round(aspect_ratios * output_height), tf.int32),
        output_width)
      crop_widths = tf.math.maximum(crop_widths, 1)

      return shape_utils.static_or_dynamic_map_fn(
              _keep_aspect_ratio_crop_and_resize,
              elems=[bboxes, crop_widths],
              dtype=tf.float32,
              parallel_iterations=self._detection_model._parallel_iterations), crop_widths

    def _first_value_op(self):
        g = tf.Graph()
        with g.as_default():
            types = [tf.string, tf.float32, tf.string, tf.int64, tf.float32, tf.string, tf.int64]
            shapes = [None, (1, None, 4), None, None, (None, 4), None, None]
            names = ["arg_{}".format(i) for i in range(len(types))]
            placeholders = [tf.placeholder(tp, name=n, shape=sh) for tp, n, sh in zip(types, names, shapes)]
            s2i, i2s = self._init_tables(self._parameters)
            compute_eval_ops = partial(self.compute_eval_ops, string2int=s2i, int2string=i2s)
            metrics = compute_eval_ops(*placeholders)
            ops = [v[1] for v in metrics.values()]
            variables = {k: var[0] for k, var in metrics.items()}

        def run_session(sess, preds):
            sess.run([s2i.init, i2s.init])
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            for args in preds:
                sess.run(ops, feed_dict={"arg_{}:0".format(i): args[i] for i in range(len(args))})
                if self.flags.compute_only_per_example_metrics:
                    metrics = sess.run(variables)
                    pprint(metrics)
                    eval_vars = [v for v in tf.local_variables() if 'evaluation/' in v.name]
                    sess.run(tf.variables_initializer(eval_vars))

            preds.clear()

        with tf.Session(graph=g) as session:
            print("Evaluting Main Stream")
            run_session(session, self._target_predictions)
            self._metrics = session.run(variables)

        with tf.Session(graph=g) as session:
            print("Evaluating Secondary Stream")
            run_session(session, self._source_predictions)
            synth_result = session.run(variables)
            self._metrics.update({k + "/synth": v for k, v in synth_result.items()})

        pprint(self._metrics)

        return self._metrics[self._metric_names[0]]

    def _predict_lstm(self, rpn_features_to_crop, detection_boxes, matched_transcriptions,
        detection_scores, detection_corpora, num_matched_detections, mode):
        """ The inner logic of CRNN. First perform crop_feature_map() to get
            field features. Then reshape the cropped field features and forward to the
            bidirectional lstm layers.

            Args:
                rpn_features_to_crop: The last feature map layer of the feature extractor. The name comes from stage 1 (the region proposal).
                detection_boxes: The detection boxes coming from stage 2. They have been post-processed and assigned to a groundtruth object.
                    A float32 Tensor of shape [num_matched_detections, 4]. It's in normalized coordinates, following tf.image.crop_and_resize() interface.
                matched_transcriptions: A string Tensor of shape [num_matched_detections], containing the target strings for each detection.
                detection_scores
                detection_corpora
                num_matched_detections
                mode
            Returns:

        """
        detection_model = self._detection_model
        if not self._backprop_detection:
            detection_boxes = tf.stop_gradient(detection_boxes)
        if not self._backprop_feature_map:
            rpn_features_to_crop = tf.stop_gradient(rpn_features_to_crop)

        flattened_detected_feature_maps, seq_len_inputs = self.crop_feature_map(rpn_features_to_crop,
            detection_boxes)    # [batch, height, width, features]

        if self.flags.explicitely_recompute_field_features:
            orig_image = self.input_features[fields.InputDataFields.image]
            flattened_detected_feature_maps, seq_len_inputs = self.crop_feature_map_debug(orig_image, detection_boxes, [x * 16 for x in self._crop_size])
            if self.flags.dump_cropped_fields_to_image_file:
                seq_len_inputs = tf.py_func(self.write_to_file, [seq_len_inputs, flattened_detected_feature_maps, self.input_features['debug'],
                    tf.tile(tf.constant([-1], dtype=tf.int64), [tf.shape(flattened_detected_feature_maps)[0]]),
                    tf.tile(tf.constant(['$']), [tf.shape(flattened_detected_feature_maps)[0]]),
                    seq_len_inputs],
                    seq_len_inputs.dtype)
            with tf.variable_scope(self._debug_root_variable_scope, reuse=True):
                flattened_detected_feature_maps, self.endpoints = (
                detection_model._feature_extractor.extract_proposal_features(
                    flattened_detected_feature_maps,
                    scope=detection_model.first_stage_feature_extractor_scope))
                seq_len_inputs = tf.cast(seq_len_inputs / 16, dtype=seq_len_inputs.dtype)

        with tf.variable_scope('Reshaping_cnn'):
            n_channels = flattened_detected_feature_maps.get_shape().as_list()[3]
            transposed = tf.transpose(flattened_detected_feature_maps, perm=[0, 2, 1, 3],
                                      name='transposed')  # [batch, width, height, features]
            conv_reshaped = tf.reshape(transposed, [-1, self._crop_size[1], self._crop_size[0] * n_channels],
                                       name='reshaped')  # [batch, width, height x features]
        transcription_dict = self.lstm_layers(conv_reshaped, detection_corpora, seq_len_inputs, mode)
        transcription_dict['labels'] = matched_transcriptions
        detections_dict = {}
        detections_dict[fields.DetectionResultFields.detection_boxes] = detection_boxes
        detections_dict[fields.DetectionResultFields.detection_scores] = detection_scores
        detections_dict[fields.DetectionResultFields.detection_corpora] = detection_corpora
        detections_dict[fields.DetectionResultFields.num_detections] = tf.cast(num_matched_detections, dtype=tf.float32)
        for k,v in detections_dict.items():
            detections_dict[k] = tf.expand_dims(v, axis=0)
        transcription_dict.update(detections_dict)
        return transcription_dict


    def str2code(self, labels, table_str2int=None):
        """Convert string label to code label"""
        with tf.name_scope('str2code_conversion'):
            if not table_str2int:
             table_str2int = self._table_str2int
            splitted = tf.string_split(labels, delimiter='')
            # values_int = tf.cast(tf.squeeze(tf.decode_raw(splitted.values, tf.uint8)), tf.int64) # Why the squeeze? it causes a bug
            values_int = tf.reshape(tf.cast(tf.decode_raw(splitted.values, tf.uint8), tf.int64), [-1])
            # values_int = tf.Print(values_int, [tf.shape(splitted.values)], message="splitted.values", summarize=9999)
            codes = table_str2int.lookup(values_int)
            codes = tf.cast(codes, tf.int32)
            return tf.SparseTensor(splitted.indices, codes, splitted.dense_shape)

    def loss(self, predictions_dict, sparse_code_target):
        # Alphabet and codes
        seq_len_inputs = predictions_dict['seq_len_inputs']

        with tf.control_dependencies([tf.less_equal(sparse_code_target.dense_shape[1], tf.reduce_max(tf.cast(seq_len_inputs, tf.int64)))]):
            loss_ctc = tf.nn.ctc_loss(labels=sparse_code_target,
                                      inputs=predictions_dict['prob'],
                                      sequence_length=seq_len_inputs,
                                      preprocess_collapse_repeated=False,
                                      ctc_merge_repeated=True,
                                      ignore_longer_outputs_than_inputs=True,
                                      time_major=True)
            loss_ctc = tf.reduce_mean(loss_ctc)

            #array of labels length
            seq_lengths_labels = tf.bincount(tf.cast(sparse_code_target.indices[:, 0], tf.int32),
                                         minlength= tf.shape(predictions_dict['prob'])[1])
        return loss_ctc


    def write_to_file(self, dummy, dt_crops, source_id, corpora, transcriptions, true_sizes):
        # TODO: replace hard-coded path with parameter. Example of path: /notebooks/Detection/workspace/models/E2E/cer_48/zoom_in/tf1.12/predictions/
        path = '/reports/{}.tfrecord'.format(source_id)
        print("!!! Write to file: The hard coded path is", path)
        debug_writer = tf.python_io.TFRecordWriter(path)
        for i in range(dt_crops.shape[0]):
            crop = dt_crops[i]
            du.write_old_tfrecord_example(debug_writer, crop[:, :true_sizes[i]], corpora[i], transcriptions[i])
        debug_writer.close()
        return dummy

    def print_string_tensor(source, dest, mess, summar=9999):
        tens = tf.concat([tf.expand_dims(x, axis=1) for x in [source, dest]], axis=1)
        return tf.map_fn(lambda t: tf.Print(t[0],[*tf.split(t, 2, axis=0), tf.shape(tens)[0]], message=mess, summarize=summar), tens)

    def compute_eval_ops(self, words, detection_boxes, matched_transcriptions, dt_positive_indices, gt_boxes, gt_transcriptions,
        debug_corpora=None, string2int=None, int2string=None):
        """ TODO: Update comments
            Compute Precision, Recall and Character Error Rate (CER).

            All metrics are backed by tf.accuracy. We devise matchings to compute the metrics.
            Matches in tf.accuracy always equal the amount of true positives. False positives and negatives
            are computed as below.

            For precision, we use the postprocessed transcriptions from the model and their assigned groundtruth
            texts. Unassigned predictions are mapped to the single code "-1", so that they will always
            be unmatched under tf.accuracy. In this case the count of unmatched rows in tf.accuracy
            equals the amount of false positives.

            For recall, we start from the perfect matching between groundtruth and predictions. Then, we pad
            it with unmatched string pairs such that the size equals the amount of groundtruth objects.
            This way, the number of unmatched rows under tf.accuracy equals the number of false negatives.

            For CER, we simply encode in sparse format the perfect matching.
        """
        if not string2int:
            string2int = self._table_str2int
        if not int2string:
            int2string = self._table_int2str

        with tf.name_scope('evaluation'):
            def encode_groundtruth(matched_transcriptions):
                matched_codes = self.str2code(matched_transcriptions, string2int)
                seq_lengths_labels = tf.bincount(tf.cast(matched_codes.indices[:, 0], tf.int32), #array of labels length
                                 minlength= tf.shape(matched_transcriptions)[0])
                target_chars = int2string.lookup(tf.cast(matched_codes, tf.int64))
                return get_words_from_chars(target_chars.values, seq_lengths_labels)

            gt_boxlist = box_list.BoxList(gt_boxes)
            all_predictions = words[0]

            # Compute Precision
            target_words = encode_groundtruth(matched_transcriptions)
            if self.flags.metrics_verbose:
                all_predictions = print_string_tensor(all_predictions, target_words, "precision")
            precision, precision_op = tf.metrics.accuracy(target_words, all_predictions,
                name='precision')


            # Compute Recall
            detection_boxlist = box_list.BoxList(detection_boxes[0])
            (_, _, _, _, match) = self._target_assigner.assign(gt_boxlist, detection_boxlist)
            padded_best_predictions = match.gather_based_on_match(all_predictions, self.NULL, self.NULL)

            unpadded_gt_transcriptions = gt_transcriptions[:gt_boxlist.num_boxes()]
            target_words = encode_groundtruth(unpadded_gt_transcriptions)
            if self.flags.metric_verbose:
                target_words = print_string_tensor(target_words, padded_best_predictions, "recall")
            recall, recall_op = tf.metrics.accuracy(target_words, padded_best_predictions,
                name='recall')

            # Compute Character Error Rate
            indicator = match.matched_column_indicator()
            sampled_matched_transcriptions = tf.boolean_mask(unpadded_gt_transcriptions, indicator)
            sampled_matched_predictions = tf.boolean_mask(padded_best_predictions, indicator)

            # This code was used to compare this architecture to the 2-staged one
            if self.flags.dump_metrics_input_to_tfrecord or self._dump_metrics_input_to_tfrecord_using_groundtruth:
                padded_dets_boxes = match.gather_based_on_match(detection_boxlist.get(), [-1.0] * 4, [-1.0] * 4)
                padded_dets_corpora = match.gather_based_on_match(debug_corpora, -2, -2)
                best_dets_boxes = tf.boolean_mask(padded_dets_boxes, indicator)
                matched_gt_boxes = tf.boolean_mask(gt_boxlist.get(), indicator)
                dets_corpora = tf.boolean_mask(padded_dets_corpora, indicator)
                img = self.input_features[fields.InputDataFields.image]
                if self.flags.dump_metrics_input_to_tfrecord_using_groundtruth:
                    best_dets_boxes = matched_gt_boxes
                crops, true_sizes = self.crop_feature_map_debug(img, best_dets_boxes, [32, 256])
                sampled_matched_predictions = tf.py_func(self.write_to_file, [sampled_matched_predictions, crops,
                    self.input_features['debug'], dets_corpora, sampled_matched_transcriptions, true_sizes],
                        sampled_matched_predictions.dtype)

            # Compute CER on non-empty vectors
            sampled_matched_transcriptions = tf.cond(tf.shape(sampled_matched_transcriptions)[0] < 1,
                lambda: tf.constant([self.NULL], dtype=tf.string),
                lambda: sampled_matched_transcriptions)
            sampled_matched_predictions = tf.cond(tf.shape(sampled_matched_predictions)[0] < 1,
                lambda: tf.constant([self.NULL], dtype=tf.string),
                lambda: sampled_matched_predictions)

            sparse_code_target = self.str2code(sampled_matched_transcriptions, string2int)
            sparse_code_pred = self.str2code(sampled_matched_predictions, string2int)
            db0 = [sparse_code_pred.indices, sparse_code_pred.values, sparse_code_pred.dense_shape]
            db = [sparse_code_target.indices, sparse_code_target.values, sparse_code_target.dense_shape]
            CER, CER_op = tf.metrics.mean(tf.edit_distance(tf.cast(sparse_code_pred, dtype=tf.int64),
                 tf.cast(sparse_code_target, dtype=tf.int64)), name='CER')

            eval_metric_ops = {
                'eval/precision' : (precision, precision_op),
                'eval/recall' : (recall, recall_op),
                'eval/CER' : (CER, CER_op)
            }
            return eval_metric_ops

    def lstm_layers(self, feature_maps, corpus, seq_len_inputs, mode):
        parameters = self._parameters

        logprob, raw_pred = deep_bidirectional_lstm(feature_maps, corpus, params=parameters, summaries=False)
        predictions_dict = {'prob': logprob,
                            'raw_predictions': raw_pred,
                            'seq_len_inputs': seq_len_inputs
                            }

        with tf.name_scope('code2str_conversion'):
            table_int2str = self._table_int2str
            sparse_code_pred, log_probability = tf.nn.ctc_beam_search_decoder(predictions_dict['prob'],
                                                                              sequence_length=seq_len_inputs,
                                                                              merge_repeated=False,
                                                                              beam_width=100,
                                                                              top_paths=parameters.nb_logprob)

            predictions_dict[fields.TranscriptionResultFields.score] = log_probability

            sequence_lengths_pred = [tf.bincount(tf.cast(sparse_code_pred[i].indices[:, 0], tf.int32),
                                                minlength=tf.shape(predictions_dict['prob'])[1]) for i in range(parameters.top_paths)]

            pred_chars = [table_int2str.lookup(sparse_code_pred[i]) for i in range(parameters.top_paths)]

            list_preds = [get_words_from_chars(pred_chars[i].values, sequence_lengths=sequence_lengths_pred[i])
                          for i in range(parameters.top_paths)]

            predictions_dict[fields.TranscriptionResultFields.words] = tf.stack(list_preds)

        return predictions_dict

