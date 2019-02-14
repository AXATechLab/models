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

import numpy as np
sys.path.append("/notebooks/text-renderer/generation/")
from pprint import pprint
from pathlib import Path
from form import TranscriptionRecordsWriter, Form

import os

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
        dump_metrics_input_to_tfrecord_using_groundtruth=False,
        dump_directory="dumps/"):
        """ Object initialization.
        Args:
            replace_detections_with_groundtruth: Train the CRNN using the groundtruth boxes instead of the
                detection boxes. The purpose of this operation is rule out the error propagating from
                the detection model in the training of CRNN. No effect on EVAL or PREDICT.
                Cannot set it on together with train_on_detections_and_groundtruth.
            train_on_detections_and_groundtruth: Train the CRNN using both the groundtruth boxes and
                the detection boxes. The purpose of this operation is to augment the data observed by CRNN
                at every step, thus speeding up training. Cannot set it on together with
                replace_detections_with_groundtruth. No effect on EVAL or PREDICT.
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
                through the feature extractor. In contrast to dump_metrics_input_to_tfrecord, the crops are real
                detections coming from stage 2.
            metrics_verbose: Enabling this flag will print to the console the arguments used to compute the metrics for every
                field (usually couples of predictions and groundtruth).
            dump_metrics_input_to_tfrecord: Dump to a tfrecord every detected field that is used to compute metrics.
                Note: in contrast to dump_cropped_fields_to_image_file, only detected fields that are actually used during
                metric computation are dumped. This means that the dumped crops are actually the highest IoU and confidence
                ones per each groundtruth object. Of course, there is no need to set explicitely_recompute_field_features.
                This is useful for comparing with other architectures using the same metrics.
            dump_metrics_input_to_tfrecord_using_groundtruth: Dump to a tfrecord every groundtruth field (image) that is used
                to compute metrics. This is useful for comparing with other architectures.
        """
        if replace_detections_with_groundtruth and train_on_detections_and_groundtruth:
            raise ValueError("""Invalid values for CRNN flags: cannot set replace_detections_with_groundtruth
                and train_on_detections_and_groundtruth both to TRUE""")
        if dump_cropped_fields_to_image_file and not explicitely_recompute_field_features:
            raise ValueError("""You need to set explicitely_recompute_field_features to TRUE to be able to
                use dump_cropped_fields_to_image_file option.""")
        self.replace_detections_with_groundtruth = replace_detections_with_groundtruth
        self.train_on_detections_and_groundtruth = train_on_detections_and_groundtruth
        self.compute_only_per_example_metrics = compute_only_per_example_metrics
        self.explicitely_recompute_field_features = explicitely_recompute_field_features
        self.dump_cropped_fields_to_image_file = dump_cropped_fields_to_image_file
        self.metrics_verbose = metrics_verbose
        self.dump_metrics_input_to_tfrecord = dump_metrics_input_to_tfrecord
        self.dump_metrics_input_to_tfrecord_using_groundtruth = dump_metrics_input_to_tfrecord_using_groundtruth
        self.dump_directory = dump_directory

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

    Details on two-streams evaluation implementation:

    In order to evaluate the SYNTH and REAL datasets, we need to separate the two from the
    input pipeline (we can only have one eval spec, so the input pipeline has to combine the two
    datasets into one). In order to do this, the input pipeline keeps track of whether the read
    image is SYNTH or REAL. More precisely, we use a different notation and call them source and
    target respectively, since no specific concept of neither 'synth' nor 'real' is used and
    any coupled dataset can be given as sorce and target (the terminology comes from domain
    adaptation). The variable that keeps the stream type is self.input_features['metrics_on_dual'].

    The Tensorflow estimator executes the update op of every metric for every example and calls
    the variable read at the end of the input stream. A naive approach would be to condition both
    operations on input_features['metrics_on_dual']. However, because of this conditioning, the variable read could
    only return one value (depending on the condition), since it is requested only once.
    With this approach, it is not possible to retrieve both metric values. Note that
    the variable read does need to be the one under the condition because tensorflow forbids any other access
    to prevent bugs.

    My solution mimics the implementation of coco metrics: the update op merely stores the parameters
    needed for each example evaluation. We maintain two datastructures, one for source and one for target,
    which are filled according to input_features['metrics_on_dual'].
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
    def _build_tables(self, parameters):
        """ Get the alphabet hash tables. This function is called when initializing a session
        for either train or eval streams.

        Args:
            parameters: A Params object, a dictionary with the following keys must be provided:
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
        self._table_str2int, self._table_int2str = self._build_tables(parameters)

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
            'eval/precision/dual' : (tf.constant(0, dtype=tf.float32), tf.no_op()),
            'eval/recall/dual' : (tf.constant(0, dtype=tf.float32), tf.no_op()),
            'eval/CER/dual' : (tf.constant(0, dtype=tf.float32), tf.no_op())
        }

        # Arbitrary ordering of the metrics. The ordering is used to determine which is the first metric to be evaluated in the dual-stream eval.
        self._metric_names = [
            'eval/precision',
            'eval/recall',
            'eval/CER',
            'eval/precision/dual',
            'eval/recall/dual',
            'eval/CER/dual',
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
            given detections in it. Structure: [loss, transcriptions_dict, eval_metric_ops].
            See predict() for details.
        """
        return lambda : self.no_forward_pass(detections_dict) + [self.no_eval_op]

    def no_forward_pass(self, detections_dict):
        """Build a placeholder forward pass of CRNN.

        Args:
            detections_dict: the postprocessed detections to be integrated in the placeholder result
        Returns:
            A placeholder comforming to a CRNN forward pass as far as type goes and with the
            given detections in it. Structure: [loss, transcriptions_dict]. See predict() for details.
        """
        transcriptions_dict = {
            'raw_predictions': tf.constant(0, dtype=tf.int64),
            'labels': tf.constant('', dtype=tf.string),
            'sequence_lengths': 0,
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
            A list of [loss, transcriptions_dict, eval_metric_ops], where any or all of those
            values could be a placeholder depending on the mode and run-time conditions. Failing a
            condition prevents the lstm layers from running.

            Thus, depending on the mode:
                1) on TRAIN: only 'loss' is valid. Moreover, there are two run-time checks.
                    One is testing whether the global step is bigger than or equal to self._start_at_step.
                    The other controls that there is at least one detection box coming from stage 2
                    that has a target groundtruth object, assigned by IoU overlap. This last check is
                    required by the ctc_loss.
                2) on EVAL: only 'transcriptions_dict' and 'eval_metric_ops' results are valid. There is only
                    one run-time check, which controls whether there is at least one detection surviving the
                    non-max suppression of stage 2 (due to the initial score thresholding).
                3) on PREDICT: only 'transcriptions_dict' is valid. No run-time checks are performed.

            Details on return values:
                1) loss: A scalar float32 tensor.
                2) transcriptions_dict: A dictionary:
                    'raw_predictions': raw predictions for debug inspection,
                    'labels': A string Tensor of groundtruth target labels for each detection and shape: [num_detections],
                    'sequence_lengths': An int64 Tensor with the length of each unpadded sequence for debug inspection. Shape: [num_detections],
                    'prob': for debug inspection,
                    'score': A float32 Tensor with confidence score for each transcription. Shape: [num_detections],
                    'words': list of length self._top_paths with string Tensors constaining the top transcriptions
                        for each detection. Tensor shape: [num_detections]
                    'detection_boxes' : A float32 Tensor of shape [num_detections, 4].
                        These are post-processed detection boxes in normalized coordinates from stage 2,
                    'detection_scores' : A float32 Tensor of shape [num_detections]. The scores of detection boxes,
                    'detection_corpora' : An int32 Tensor of shape [num_detections]. The corpus type assigned to detection boxes,
                    'num_detections' : A scalar int64 Tensor, the number of detections.
                3) eval_metric_ops: A float32 metric dictionary:
                    'eval/precision': precision for source stream,
                    'eval/recall': recall for source stream,
                    'eval/CER': CER for source stream,
                    'eval/precision/synth': precision for source stream,
                    'eval/recall/synth': recall for target stream,
                    'eval/CER/synth': CER for target stream,
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

    def _fetch_groundtruth(self, true_image_shapes):
        """ Fetch groundtruth. We maintain both normalized and absolute groundtruth
        according to the need. If the input has no groundtruth object, the result is
        empty boxlists and tensors.

        Args:
            true_image_shapes: An int64 Tensor of shape [1, H, W, C].
        Returns:
`           A BoxList of length [groundtruth_size] in normalized coordinates.
            A BoxList of length [groundtruth_size] in absolute coordinates.
            A string Tensor of shape [groundtruth_size] containing groundtruth labels.
        """
        groundtruth_boxlists, _, _, _, groundtruth_text = self._detection_model._format_groundtruth_data(true_image_shapes,
            stage='transcription')
        normalized_groundtruth_boxlist = box_list.BoxList(tf.placeholder(shape=(1, 4), dtype=tf.float32))
        normalize_groundtruth_fn = lambda: box_list_ops.to_normalized_coordinates(groundtruth_boxlists[0],
            true_image_shapes[0, 0], true_image_shapes[0, 1]).get()
        # Guard for examples with no objects to detect (box_list_ops throws an exception)
        normalized_groundtruth_boxlist.set(tf.cond(groundtruth_boxlists[0].num_boxes() > 0, normalize_groundtruth_fn,
            lambda: groundtruth_boxlists[0].get()))
        return normalized_groundtruth_boxlist, groundtruth_boxlists[0], groundtruth_text[0]

    def _assign_detection_targets(self, normalized_detection_boxlist, groundtruth_boxlist, groundtruth_text, true_image_shapes, mode):
        """ Find the template and groundtruth target for each detection. A detection's target is the box that has
        the highest IoU overlap with the detection box.

        TODO: investigate if assignments on normal coordinates are allowed, since this is not the way it was
        done on FasterRCNN.

        Args:
            normalized_detection_boxlist: A BoxList of detections in normalized coordinates,
            groundtruth_boxlist: A BoxList of groundtruth in absolute coordinates,
            groundtruth_text: A string Tensor of grountruth labels,
            true_image_shapes: The input image size Tensor [1, H, W, C],
            mode: The Estimator's mode (TRAIN, EVAL, PREDICT).

        Returns:
            A BoxList of detections same as normalized_detection_boxlist but:
                1) Computed corpus targets (from the template) are stored in 'corpus'.
                2) If in TRAIN or EVAL mode, computed text targets (from the groundtruth)
                    are stored in 'groundtruth_text'.
                3) If in TRAIN mode, unassigned detections are filtered out of the boxlist,
                    with their corresponding targets.
        """
        template_boxlist = box_list.BoxList(self._detection_model.current_template_boxes)
        (_, _, _, _, match) = self._template_assigner.assign(normalized_detection_boxlist, template_boxlist)
        template_corpora = self._detection_model.current_template_corpora
        zero_encoding = tf.constant(-1, dtype=tf.int32)
        assigned_detection_corpora = match.gather_based_on_match(template_corpora, zero_encoding, zero_encoding)
        normalized_detection_boxlist.add_field(fields.BoxListFields.corpus, assigned_detection_corpora)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return normalized_detection_boxlist

        detection_boxlist = box_list_ops.to_absolute_coordinates(normalized_detection_boxlist,
            true_image_shapes[0, 0], true_image_shapes[0, 1])
        (_, _, _, _, match) = self._target_assigner.assign(detection_boxlist,
            groundtruth_boxlist)
        target_words = match.gather_based_on_match(groundtruth_text, self.NULL, self.NULL)
        normalized_detection_boxlist.add_field(fields.BoxListFields.groundtruth_text, target_words)

        if mode == tf.estimator.ModeKeys.TRAIN:
            positive_indicator = match.matched_column_indicator()
            normalized_detection_boxlist = box_list_ops.boolean_mask(normalized_detection_boxlist, positive_indicator)
        return normalized_detection_boxlist

    def _build_forward_pass(self, normalized_detection_boxlist, rpn_features_to_crop, true_image_shapes, mode):
        """ Build the forward pass function, which can be run conditionally.

        Args:
            normalized_detection_boxlist: A BoxList of detections.
                'scores', 'corpus', 'groundtruth_text' are required fields,
            rpn_features_to_crop: Float32 Tensor. The last feature map layer. Shape: [1, h, w, D],
            true_image_shapes Int64 Tensor of shape [1, H, W, C],
            mode: The Estimator's mode (TRAIN, EVAL, PREDICT).

        Returns:
            The forward pass function, which returns:
                A list [loss, transcriptions_dict]. See predict() for information on this values
                according to the mode.
        """
        def forward_pass():
            normalized_detection_boxes = normalized_detection_boxlist.get()
            if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN]:
                target_words = normalized_detection_boxlist.get_field(fields.BoxListFields.groundtruth_text)
            elif mode == tf.estimator.ModeKeys.PREDICT:
                target_words = tf.constant('', dtype=tf.string)
            detection_scores = normalized_detection_boxlist.get_field(fields.BoxListFields.scores)
            detection_corpora = normalized_detection_boxlist.get_field(fields.BoxListFields.corpus)
            num_detections = normalized_detection_boxlist.num_boxes()
            transcriptions_dict = self._compute_predictions(rpn_features_to_crop, normalized_detection_boxes, target_words,
                    detection_scores, detection_corpora, num_detections)
            if mode == tf.estimator.ModeKeys.TRAIN:
                sparse_code_target = self.str2code(target_words)
                return [self.loss(transcriptions_dict, sparse_code_target), transcriptions_dict]
            elif mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
                return [self._zero_loss, transcriptions_dict]

        return forward_pass

    def _compute_eval_ops(self, normalized_detection_boxlist, normalized_groundtruth_boxlist, groundtruth_text, transcriptions_dict):
        """ Compute the third value returned by predict(): eval_metric_ops, a dictionary of metrics.

        Args:
            normalized_detection_boxlist: A BoxList of detections in normalized coordinates.
                Required fields: 'groundtruth_text', 'corpus',
            normalized_groundtruth_boxlist: A BoxList of groundtruth in normalized coordinates,
            groundtruth_text: A string Tensor of shape [padded_groundtruth_size] with padded labels,
            transcriptions_dict: See predict() for information.

        Returns:
            The eval_metric_ops dictionary. See predict().

        """
        target_words = normalized_detection_boxlist.get_field(fields.BoxListFields.groundtruth_text)
        assigned_detection_corpora = normalized_detection_boxlist.get_field(fields.BoxListFields.corpus)
        # The update ops run for every image example. They merely store the args of compute_eval_ops()
        # in python lists. There's one op per eval stream.
        source_update_op = lambda *args: self._source_predictions.append(args)
        target_update_op = lambda *args: self._target_predictions.append(args)
        common_args = [
            transcriptions_dict['words'],
            transcriptions_dict['detection_boxes'],
            target_words,
            normalized_groundtruth_boxlist.get(),
            groundtruth_text,
            assigned_detection_corpora]
        if self.flags.dump_metrics_input_to_tfrecord or self.flags.dump_metrics_input_to_tfrecord_using_groundtruth:
            common_args.append(self.input_features['filename'][0])
            common_args.append(self.input_features[fields.InputDataFields.image])
        update_op = tf.cond(self.input_features['metrics_on_dual'],
            lambda: tf.py_func(source_update_op, common_args, []),
            lambda: tf.py_func(target_update_op, common_args, []))

        # This var does the actual metric evaluation and stores the result in self._metrics
        first_var = tf.py_func(self._first_value_op, [], tf.float32)
        eval_metric_ops = {self._metric_names[0]: (first_var, update_op)}

        # Compute all metrics only once.
        with tf.control_dependencies([first_var]):
            for metric in self._metric_names[1:]:
                eval_metric_ops[metric] = (tf.py_func(lambda m: self._metrics[m.decode('latin1')], [metric], tf.float32),
                    update_op)
        return eval_metric_ops

    def _predict(self, prediction_dict, true_image_shapes, mode):
        """ Post-process stage 2 and perform forward pass, or fail.

        Args:
            prediction_dict: Raw stage 2 predictions. See FasterRCNNMetaArch.predict(),
            true_image_shapes: the shape of the input image: [1, H, W, C],
            mode: The Estimator's mode (TRAIN, EVAL, PREDICT).

        Returns:
            Same values as predict().
        """
        # Postprocess and unpad detections.
        detections_dict = self._detection_model._postprocess_box_classifier(
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
        detections_dict[fields.DetectionResultFields.detection_corpora] = tf.constant([[0]], dtype=tf.int32)
        # Remove detection classes since text detection is not a multiclass problem
        detections_dict.pop(fields.DetectionResultFields.detection_classes)
        normalized_detection_boxlist = box_list.BoxList(normalized_detection_boxes)

        normalized_groundtruth_boxlist, groundtruth_boxlist, groundtruth_text = None, None, None
        if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN]:
            normalized_groundtruth_boxlist, groundtruth_boxlist, groundtruth_text = self._fetch_groundtruth(true_image_shapes)
            if mode == tf.estimator.ModeKeys.TRAIN:
                if self.flags.replace_detections_with_groundtruth:
                    normalized_detection_boxlist = normalized_groundtruth_boxlist
                    num_detections = normalized_groundtruth_boxlist.num_boxes()
                    detection_scores = tf.placeholder(dtype=tf.float32)
                if self.flags.train_on_detections_and_groundtruth:
                    normalized_detection_boxlist = box_list_ops.concatenate([normalized_detection_boxlist,
                        normalized_groundtruth_boxlist])
                    num_detections = num_detections + normalized_groundtruth_boxlist.num_boxes()
                    detection_scores = tf.placeholder(dtype=tf.float32)
        normalized_detection_boxlist.add_field(fields.BoxListFields.scores, detection_scores)
        normalized_detection_boxlist = self._assign_detection_targets(normalized_detection_boxlist,
            groundtruth_boxlist, groundtruth_text, true_image_shapes, mode)

        tf.summary.scalar("Batch_Size", normalized_detection_boxlist.num_boxes())

        forward_pass = self._build_forward_pass(normalized_detection_boxlist, rpn_features_to_crop, true_image_shapes, mode)
        if mode == tf.estimator.ModeKeys.PREDICT:
            fail_cond = tf.constant(False, dtype=tf.bool)
        else:
            fail_cond = tf.equal(normalized_detection_boxlist.num_boxes(), 0)
        loss, transcriptions_dict = tf.cond(fail_cond,
                lambda : self.no_forward_pass(detections_dict), forward_pass, name='BatchCond')
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = self._compute_eval_ops(normalized_detection_boxlist, normalized_groundtruth_boxlist,
                groundtruth_text, transcriptions_dict)
        else:
            eval_metric_ops = self.no_eval_op

        return [loss, transcriptions_dict, eval_metric_ops]

    def crop_feature_map(self, features_to_crop, detection_boxes):
        """ Function that extracts field features from the last feature extractor map.
        This function is backed by tf.image.crop_and_resize.

        The main advantage with respect to tf.image.crop_and_resize is the minimal
        distortion on height and width.
        Indeed, the width of the detection_boxes is kept as-is in the final crop. The height
        is instead distorted to fixed value (self._crop_size[0]).

        The second operation this function has is to pad crops to self._crop_size[1].
        There is an exception for boxes that are longer than self._crop_size[1]. In that case
        The width is not kept the same but it's distorted to self._crop_size[1].

        Args:
            features_to_crop: The last feature map layer. A float32 Tensor of shape
                [1, h, w, D].
            detection_boxes: The detection boxes coming from stage 2, after CRNN pre-processing.
                A float32 Tensor of shape [num_detections, 4].
        Returns:
            A float 32 Tensor of shape
                [num_detections, self._crops_size[0], self._crop_size[1], D]
        """
        output_height, output_width = self._crop_size

        def _keep_width_crop_and_resize(args):
            bbox, crop_width = args
            fixed_height_crop = tf.image.crop_and_resize(features_to_crop,
              tf.expand_dims(bbox, axis=0), [0], [output_height, crop_width])
            padded_crop = tf.pad(fixed_height_crop[0],
              [[0, 0], [0, output_width - crop_width], [0, 0]], "CONSTANT")
            return padded_crop

        num_feat_map_cells = tf.cast(tf.shape(features_to_crop)[2], dtype=tf.float32) * (detection_boxes[:, 3] - detection_boxes[:, 1])
        crop_widths = tf.math.minimum(tf.cast(tf.round(2.0 * num_feat_map_cells), tf.int32),
            output_width)
        crop_widths = tf.math.maximum(crop_widths, 1)

        return shape_utils.static_or_dynamic_map_fn(
              _keep_width_crop_and_resize,
              elems=[detection_boxes, crop_widths],
              dtype=tf.float32,
              parallel_iterations=self._detection_model._parallel_iterations), crop_widths

    def crop_feature_map_keep_aspect_ratio(self, image, detection_boxes, crop_size):
        """ Function that extracts field features from the last feature extractor map.
        This function is backed by tf.image.crop_and_resize.

        The only difference with crop_feature_map() is in how the width is computed.
        In this case we keep the aspect ratio of the bounding box after changing the
        height to self._crop_size[0]. Note that crop_feature_map() has proved to work
        better in terms of sequence length with feature maps that have high resolution.

        Args:
            features_to_crop: The last feature map layer. A float32 Tensor of shape
                [1, h, w, D].
            detection_boxes: The detection boxes coming from stage 2, after CRNN pre-processing.
                A float32 Tensor of shape [num_detections, 4].
            crop_size: The target crop size. A list of rank 2 [H, W].
        Returns:
            A float 32 Tensor of shape
                [num_detections, self._crops_size[0], self._crop_size[1], D]
        """
        output_height, output_width = crop_size

        def _keep_aspect_ratio_crop_and_resize(args):
            bbox, crop_width = args
            fixed_height_crop = tf.image.crop_and_resize(image,
              tf.expand_dims(bbox, axis=0), [0], [output_height, crop_width])
            padded_crop = tf.pad(fixed_height_crop[0],
              [[0, 0], [0, output_width - crop_width], [0, 0]], "CONSTANT")
            return padded_crop

        aspect_ratios = (detection_boxes[:, 3] - detection_boxes[:, 1]) / (detection_boxes[:, 2] - detection_boxes[:, 0])
        crop_widths = tf.math.minimum(tf.cast(tf.round(aspect_ratios * output_height), tf.int32),
        output_width)
        crop_widths = tf.math.maximum(crop_widths, 1)

        return shape_utils.static_or_dynamic_map_fn(
              _keep_aspect_ratio_crop_and_resize,
              elems=[detection_boxes, crop_widths],
              dtype=tf.float32,
              parallel_iterations=self._detection_model._parallel_iterations), crop_widths

    def _run_session(self, sess, preds, variables, ops, s2i, i2s):
        """ Run metric update operations for all stored examples.

        Args:
            sess: A tf.Session() on the metric graph.
            preds: A list of numpy arrays of _build_metric_graph() arguments.
            ops: A list of update_ops to run.
            s2i: A HashTable encoder.
            i2s: A HashTable decoder.
        """
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


    def _first_value_op(self):
        """A function embedded in a tf.py_func() node, performing metric computation. It starts two sessions,
            one for each eval stream.

        Returns:
            The variable or tensor containing the value of self._metric_names[0] metric.
        """
        g = tf.Graph()
        with g.as_default():
            # Build the metric computation graph.
            types = [tf.string, tf.float32, tf.string, tf.float32, tf.string, tf.int32]
            shapes = [None, (1, None, 4), None, (None, 4), None, None]
            if self.flags.dump_metrics_input_to_tfrecord or self.flags.dump_metrics_input_to_tfrecord_using_groundtruth:
                types += [tf.string, tf.float32]
                shapes += [None, (1, None, None, None)]
            names = ["arg_{}".format(i) for i in range(len(types))]
            placeholders = [tf.placeholder(tp, name=n, shape=sh) for tp, n, sh in zip(types, names, shapes)]
            s2i, i2s = self._build_tables(self._parameters)
            _build_metric_graph = partial(self._build_metric_graph, string2int=s2i, int2string=i2s)
            metrics = _build_metric_graph(*placeholders)
            ops = [v[1] for v in metrics.values()]
            variables = {k: var[0] for k, var in metrics.items()}

        with tf.Session(graph=g) as session:
            print("Evaluating Main Stream")
            self._run_session(session, self._target_predictions, variables, ops, s2i, i2s)
            self._metrics = session.run(variables)

        with tf.Session(graph=g) as session:
            print("Evaluating Secondary Stream")
            self._run_session(session, self._source_predictions, variables, ops, s2i, i2s)
            duaL_result = session.run(variables)
            self._metrics.update({k + "/dual": v for k, v in duaL_result.items()})

        pprint(self._metrics)

        return self._metrics[self._metric_names[0]]

    def _explicitely_recompute_field_features(self, detection_boxes):
        """ This function is called through self.flags.explicitely_recompute_field_features.
        It overrides crop_feature_map() logic by explicitely cropping the original image
        and recomputing the feature map on the new image crops.

        The point of this operation is to rule out tf.image.crop_and_resize on the feature extractor
        feature map as part of the computational graph, in order to test its effect.

        Args:
            detection_boxes: A float32 Tensor of shape [num_detections, 4]. The post-processed output of stage 2.

        Returns:
            The same values as crop_feature_map().
         """
        orig_image = self.input_features[fields.InputDataFields.image]
        downscale = self._detection_model._feature_extractor._first_stage_features_stride
        # Here we don't use crop_feature_map() because there is no reason to keep the width
        # of detection unchanged.
        flattened_detected_feature_maps, sequence_lengths = self.crop_feature_map_keep_aspect_ratio(orig_image, detection_boxes,
            [x * downscale for x in self._crop_size])
        if self.flags.dump_cropped_fields_to_image_file:
            sequence_lengths = tf.py_func(self.write_to_file, [sequence_lengths, flattened_detected_feature_maps,
                self.input_features['filename'][0],
                tf.tile(tf.constant([-1], dtype=tf.int64), [tf.shape(flattened_detected_feature_maps)[0]]),
                tf.tile(tf.constant(['$']), [tf.shape(flattened_detected_feature_maps)[0]]),
                sequence_lengths,
                os.path.join(self.flags.dump_directory, "detected_fields")],
                sequence_lengths.dtype)
        with tf.variable_scope(self._debug_root_variable_scope, reuse=True):
            flattened_detected_feature_maps, self.endpoints = (
            self._detection_model._feature_extractor.extract_proposal_features(
                flattened_detected_feature_maps,
                scope=self._detection_model.first_stage_feature_extractor_scope))
            sequence_lengths = tf.cast(sequence_lengths / downscale, dtype=sequence_lengths.dtype)
        return flattened_detected_feature_maps, sequence_lengths

    def _compute_predictions(self, rpn_features_to_crop, detection_boxes, target_words,
        detection_scores, detection_corpora, num_detections):
        """ The inner logic of CRNN. First perform crop_feature_map() to get
            field features. Then reshape the cropped field features and forward to the
            bidirectional lstm layers.

            Args:
                rpn_features_to_crop: The last feature map layer of the feature extractor. The name comes from stage 1 (the region proposal).
                    A float32 Tensor of shape [1, h, w, D].
                detection_boxes: The detection boxes coming from stage 2. They have been post-processed and assigned to a groundtruth object.
                    A float32 Tensor of shape [num_detections, 4]. It's in normalized coordinates, following tf.image.crop_and_resize() interface.
                target_words: A string Tensor of shape [num_detections], containing the target strings for each detection.
                detection_scores: A float32 Tensor of shape [num_detections], containing the confidence score of each detection.
                    Not passed to the LSTM network. Just used to build transcriptions_dict.
                detection_corpora: An int32 Tensor of shape [num_detections]. Stores the assigned type to each detection.
                num_detections: A scalar int64 Tensor. The number of detection boxes coming from stage 2. Note that there usually is pre-processing before calling
                    this function, therefore (in TRAIN mode) this is the number of detections that have a transcription target.
            Returns:
                The transcriptions_dict. See predict() for more information.
        """
        if not self._backprop_detection:
            detection_boxes = tf.stop_gradient(detection_boxes)
        if not self._backprop_feature_map:
            rpn_features_to_crop = tf.stop_gradient(rpn_features_to_crop)

        # flattened_detected_feature_maps.shape = [num_detections, self._crop_size[0], self._crop_size[1], D]
        if self.flags.explicitely_recompute_field_features:
            flattened_detected_feature_maps, sequence_lengths = self._explicitely_recompute_field_features(detection_boxes)
        else:
            flattened_detected_feature_maps, sequence_lengths = self.crop_feature_map(rpn_features_to_crop,
                detection_boxes)

        with tf.variable_scope('Reshaping_cnn'):
            n_channels = flattened_detected_feature_maps.get_shape().as_list()[3]
            transposed = tf.transpose(flattened_detected_feature_maps, perm=[0, 2, 1, 3],
                                      name='transposed')  # [batch, width, height, features]
            conv_reshaped = tf.reshape(transposed, [-1, self._crop_size[1], self._crop_size[0] * n_channels],
                                       name='reshaped')  # [batch, width, height x features]
        transcriptions_dict = self._lstm_layers(conv_reshaped, detection_corpora, sequence_lengths)
        transcriptions_dict['labels'] = target_words
        detections_dict = {}
        detections_dict[fields.DetectionResultFields.detection_boxes] = detection_boxes
        detections_dict[fields.DetectionResultFields.detection_scores] = detection_scores
        detections_dict[fields.DetectionResultFields.detection_corpora] = detection_corpora
        detections_dict[fields.DetectionResultFields.num_detections] = tf.cast(num_detections, dtype=tf.float32)
        for k,v in detections_dict.items():
            detections_dict[k] = tf.expand_dims(v, axis=0)
        transcriptions_dict.update(detections_dict)
        return transcriptions_dict


    def str2code(self, labels, table_str2int=None):
        """Convert string label to code label

        Args:
            labels: A string Tensor of shape [groundtruth_size].
            table_str2int: HashTable encoder. If None defaults to
                self._table_str2int
        Returns:
            A SparseTensor, the encoded version of labels.
        """
        with tf.name_scope('str2code_conversion'):
            if not table_str2int:
             table_str2int = self._table_str2int
            splitted = tf.string_split(labels, delimiter='')
            values_int = tf.reshape(tf.cast(tf.decode_raw(splitted.values, tf.uint8), tf.int64), [-1])
            codes = table_str2int.lookup(values_int)
            codes = tf.cast(codes, tf.int32)
            return tf.SparseTensor(splitted.indices, codes, splitted.dense_shape)

    def loss(self, transcriptions_dict, sparse_code_target):
        """The ctc loss.

        Args:
            transcriptions_dict: See predict() for details.
            sparse_code_target: A sparse Tensor containing encoded groundtruth.

        Returns:
            A scalar float32 Tensor.
        """
        # Alphabet and codes
        sequence_lengths = transcriptions_dict['sequence_lengths']

        with tf.control_dependencies([tf.less_equal(sparse_code_target.dense_shape[1], tf.reduce_max(tf.cast(sequence_lengths, tf.int64)))]):
            loss_ctc = tf.nn.ctc_loss(labels=sparse_code_target,
                                      inputs=transcriptions_dict['prob'],
                                      sequence_length=sequence_lengths,
                                      preprocess_collapse_repeated=False,
                                      ctc_merge_repeated=True,
                                      ignore_longer_outputs_than_inputs=True,
                                      time_major=True)
            loss_ctc = tf.reduce_mean(loss_ctc)

            #array of labels length
            seq_lengths_labels = tf.bincount(tf.cast(sparse_code_target.indices[:, 0], tf.int32),
                                         minlength= tf.shape(transcriptions_dict['prob'])[1])
        return loss_ctc


    def write_to_file(self, x, img, dt_boxes, transcriptions, source_id, corpora, gt_text, path):
        """ Debug function to dump image tensors to file. To embed in a tf.py_func() node.

        Args:
            x: This function behaves like tf.identity(). This array is returned with
                no changes.
            dt_crops: A float32 numpy image of shape [B, H, W, C].
            source_id: The source id of the original input image. A scalar string numpy arrray.
            corpora: An int32 numpy array of shape [B]. The corpus types of the images.
            transcriptions: A string numpy array of shape [B]. The groundtruth content of the image Tensors.
            true_sizes: An int64 numpy array of shape [B]. The unpadded widths of the images.

        Returns:
            A numpy array identical to x.
        """
        path = os.path.join(path.decode("latin1"), source_id.decode('latin1'))
        writer = TranscriptionRecordsWriter(Path(path))
        # for i in range(dt_crops.shape[0]):
        #     crop = dt_crops[i]
        #     f = Form(crop)
        #     f.boxes = np.array([[0, 0, f.height, true_sizes[i]]])
        #     f.type_codes = np.array([corpora[i]])
        #     f.transcriptions = np.array([gt_text[i].decode('latin1')])
        #     writer.write(f)
        labels = np.empty(transcriptions.shape[0])
        for i in range(transcriptions.shape[0]):
            labels[i] = "{gt}|{prediction}".format(gt=gt_text[i].decode('latin1'),
                prediction=transcriptions[i].decode('latin1'))
        f = Form(img)
        f.boxes = dt_boxes
        f.type_codes = corpora
        f.transcriptions = labels
        writer.write(f)
        writer.close()
        return x

    def print_compare_string_tensors(self, source, dest, mess, summar=9999):
        """Print two string tensors side by side to the console.

        Args:
            source: A string Tensor of shape [N].
            dest: A string Tensor of shape [N].
            mess: A string message.

        Returns:
            A Tensor identical to the source.
        """
        tens = tf.concat([tf.expand_dims(x, axis=1) for x in [source, dest]], axis=1)
        return tf.map_fn(lambda t: tf.Print(t[0],[*tf.split(t, 2, axis=0), tf.shape(tens)[0]], message=mess, summarize=summar), tens)

    def _re_encode_groundtruth(self, matched_transcriptions, string2int, int2string):
        """Re-encode groundtruth. TODO: maybe remove this step."""
        matched_codes = self.str2code(matched_transcriptions, string2int)
        # array of labels length
        seq_lengths_labels = tf.bincount(tf.cast(matched_codes.indices[:, 0], tf.int32),
                         minlength= tf.shape(matched_transcriptions)[0])
        target_chars = int2string.lookup(tf.cast(matched_codes, tf.int64))
        return get_words_from_chars(target_chars.values, seq_lengths_labels)

    def assign_top_words_to_groundtruth(self, groundtruth_boxlist, detection_boxlist):
        """ Assign the best prediction to every groundtruth object.
        Args:
            groundtruth_boxlist: A BoxList of size groundtruth_size.
            detection_boxlist: A BoxList of size num_detections. We expect that 'transcription'
                field is filled with top predictions for each detection box.
        Returns:
            A Match object encoding the assignment of detections to groundtruth. Moreover,
            groundtruth_boxlist's transcription field is filled with the best prediction per
            groundtruth object.
        """
        top_words = detection_boxlist.get_field(fields.BoxListFields.transcription)
        (_, _, _, _, match) = self._target_assigner.assign(groundtruth_boxlist, detection_boxlist)
        assigned_top_words = match.gather_based_on_match(top_words, self.NULL, self.NULL)
        groundtruth_boxlist.add_field(fields.BoxListFields.transcription, assigned_top_words)
        return match

    def compute_precision(self, words, target_words, string2int, int2string):
        """ Compute precision, i.e. the proportion of detection boxes whose transcription matches the groundtruth. This metric is quite
        coarse-grained, in that it measures performance at field level. For a finer granularity metric see compute_CER(),
        which measures accuracy at character level.

        Details on how this is computed:
            We rely on tf.metrics.accuracy, giving as input the highest confidence transcription per detection and the corresponding vector of
            target text (assigned according to highest IoU). tf.metrics.accuracy computes
            the ratio of matching vector rows over the size of the vector. The size of the vector is the same
            as num_detections. A matching row is equivalent to a True Positive. An unmatched row is equivalent
            to a False Positive (or a False Negative). Therefore, here we are computing matched_rows / (matched_rows +
            + unmatched_rows) = True Positives / (True Positives + False Positives). Since we include in this computation
            all True Positives and all False Positives, this quantity is indeed precision.

        Args:
            words: A string Tensor of shape [num_detections] containing top predictions.
            target_words: A string Tensor of shape [num_detections] containing assigned groundtruth objects.
            string2int: A HashTable encoder.
            int2string: A HashTable decoder.

        Returns:
            variable and update_op for precision.
        """
        target_words = self._re_encode_groundtruth(target_words, string2int, int2string)
        if self.flags.metrics_verbose:
            words = self.print_compare_string_tensors(words, target_words, "precision")
        return tf.metrics.accuracy(words, target_words, name='precision')

    def compute_recall(self, groundtruth_boxlist, string2int, int2string, detection_boxlist=None, match=None):
        """ Compute recall, i.e. the proportion of groundtruth text that are correctly transcribed. This metric is quite
        coarse-grained, in that it measures performance at field level. For a finer granularity metric see compute_CER(),
        which measures accuracy at character level.

        This function can be interfaced in two different ways: either by providing a Match object encoding the assignment
        of detections to groundtruth, or by explicitely providing the boxlist of detections. In this last case the match
        will be computed internally.

        Details on how this is computed:
            We rely on tf.metrics.accuracy, giving as input the groundtruth text vector and the corresponding vector of
            best model predictions (best in terms of transcription confidence and IoU). tf.metrics.accuracy computes
            the ratio of matching vector rows over the size of the vector. The size of the vector is the same
            as groundtruth_size. A matching row is equivalent to a True Positive. An unmatched row is equivalent
            to a False Negative (or a False Positive). Therefore, here we are computing matched_rows / (matched_rows +
            + unmatched_rows) = True Positives / (True Positives + False Negatives). Since we include in this computation
            all True Positives and all False Negatives, this quantity is indeed recall.

        Args:
            groundtruth_boxlist: A BoxList with groundtruth. The field 'groundtruth_text' is required.
                groundtruth_text is a string Tensor of shape [groundtruth_boxlist.num_boxes()].
                If match argument is provided, 'transcription' field is also required.
            string2int: A HashTable encoder.
            int2string: A HashTable decoder.
            detection_boxlist: A BoxList with detections. The field 'transcription' is required.
                'transcription' is a string Tensor of shape [detection_boxlist.num_boxes()].
                If match argument is also provided, this argument is unused.
            match: A Match object encoding assignemnt of detections to groundtruth. If not provided, then
                detection_boxlist MUST be passed in. This will compute the matching internally.

        Returns:
            variable, update_op for recall and a Match object encoding the assignment of groundtruth boxes to
            detection boxes. If match argument is provided, this is that same match object.
            The field 'transcription' of groundtruth_boxlist is filled with groundtruth assigned top
            predictions.
        """
        if not match and not detection_boxlist:
            raise ValueError("Cannot compute recall: both detection_boxlist and match object were not provided")
        if not match:
            match = self.assign_top_words_to_groundtruth(groundtruth_boxlist, detection_boxlist)

        groundtruth_text = groundtruth_boxlist.get_field(fields.BoxListFields.groundtruth_text)
        assigned_top_words = groundtruth_boxlist.get_field(fields.BoxListFields.transcription)
        groundtruth_text = self._re_encode_groundtruth(groundtruth_text, string2int, int2string)
        if self.flags.metrics_verbose:
            groundtruth_text = self.print_compare_string_tensors(groundtruth_text, assigned_top_words, "recall")
        recall, recall_op = tf.metrics.accuracy(groundtruth_text, assigned_top_words, name='recall')
        return recall, recall_op, match

    def compute_CER(self, groundtruth_boxlist, string2int, detection_boxlist=None, match=None, dump_fn=None):
        """ Compute Character Error Rate (CER) metric. This metric is expected to be computed between target words and
        assigned predicted words whose detection box has the highest IoU with the groundtruth box. Indeed, we don't want
        to compute CER on all detections, where some of them may have no targets or might easily be of bad quality.

        In case of no possible matching between detections
        and groundtruth, as in the case of an empty image, CER cannot be computed and we return 0.0 .

        This function can be interfaced in two different ways: either by providing a Match object encoding the assignment
        of detections to groundtruth, or by explicitely providing the boxlist of detections. In this last case the match
        will be computed internally.

        Args:
            groundtruth_boxlist: A BoxList with groundtruth. The field 'groundtruth_text' is required.
                groundtruth_text is a string Tensor of shape [groundtruth_boxlist.num_boxes()].
                If match argument is provided, 'transcription' field is also required.
            string2int: A HashTable encoder.
            detection_boxlist: A BoxList with detections. The field 'transcription' is required.
                'transcription' is a string Tensor of shape [detection_boxlist.num_boxes()].
                If match argument is also provided, this argument is unused.
            match: A Match object encoding assignemnt of detections to groundtruth. If not provided, then
                detection_boxlist MUST be passed in. This will compute the matching internally.
            dump_fn: An optional function for dumping input to a tfrecord.
                Args:
                    x: A Tensor used to hook the function to the graph.
                    matched_groundtruth_boxlist: A BoxList of size matched_groundtruth_size.
                        'groundtruth_text' and 'transcription' fields are filled in.
                    match: A Match object encoding assignment of detections to groundtruth.
                Returns:
                    The Tensor x.
        Returns:
            variable, update_op for CER and a Match object encoding the assignment of groundtruth boxes to
            detection boxes. If match argument is provided, this is that same match object.
            The field 'transcription' of groundtruth_boxlist is filled with groundtruth assigned top
            predictions.

        """
        if not match and not detection_boxlist:
            raise ValueError("Cannot compute CER: both detection_boxlist and match object were not provided")
        if not match:
            match = self.assign_top_words_to_groundtruth(groundtruth_boxlist, detection_boxlist)

        indicator = match.matched_column_indicator()
        matched_groundtruth_boxlist = box_list_ops.boolean_mask(groundtruth_boxlist, indicator)
        matched_groundtruth_text = matched_groundtruth_boxlist.get_field(fields.BoxListFields.groundtruth_text)
        matched_predicted_text = matched_groundtruth_boxlist.get_field(fields.BoxListFields.transcription)
        if dump_fn:
            matched_predicted_text = dump_fn(matched_predicted_text, matched_groundtruth_boxlist, match)
        # Compute CER on non-empty vectors. Since the two vectors are matched, they are both either empty or non-empty.
        # In the former case, we insert a null character in both vectors and compute CER so that it returns 0.0 .
        matched_predicted_text = tf.cond(tf.shape(matched_predicted_text)[0] < 1,
            lambda: tf.constant([self.NULL], dtype=tf.string),
            lambda: matched_predicted_text)
        matched_groundtruth_text = tf.cond(tf.shape(matched_groundtruth_text)[0] < 1,
            lambda: tf.constant([self.NULL], dtype=tf.string),
            lambda: matched_groundtruth_text)
        sparse_code_pred = self.str2code(matched_predicted_text, string2int)
        sparse_code_target = self.str2code(matched_groundtruth_text, string2int)
        CER, CER_op = tf.metrics.mean(tf.edit_distance(tf.cast(sparse_code_pred, dtype=tf.int64),
             tf.cast(sparse_code_target, dtype=tf.int64)), name='CER')
        return CER, CER_op, match

    def dump_tfrecord(self, x, matched_groundtruth_boxlist, match, detection_boxlist, debug_corpora, filename, image):
        """ This function was used to compare this architecture to the 2-staged one
        Args:
            x: A Tensor used to hook the function to the graph.
            matched_groundtruth_boxlist: A BoxList of size matched_groundtruth_size.
                'groundtruth_text' and 'transcription' fields are filled in.
            match: A Match object encoding assignment of detections to groundtruth.
            detection_boxlist: A BoxList of size num_detections.
            debug_corpora: An int32 Tensor of shape [num_detections].
        Returns:
            The Tensor x.
        """
        assigned_detection_boxes = match.gather_based_on_match(detection_boxlist.get(), [-1.0] * 4, [-1.0] * 4)
        assigned_detection_corpora = match.gather_based_on_match(debug_corpora, -2, -2)
        indicator = match.matched_column_indicator()
        matched_detection_boxes = tf.boolean_mask(assigned_detection_boxes, indicator)
        matched_detection_corpora = tf.boolean_mask(assigned_detection_corpora, indicator)
        matched_transcriptions = matched_groundtruth_boxlist.get_field(fields.BoxListFields.transcription)
        matched_groundtruth_text = matched_groundtruth_boxlist.get_field(fields.BoxListFields.groundtruth_text)
        if self.flags.dump_metrics_input_to_tfrecord_using_groundtruth:
            crop_boxes = matched_groundtruth_boxlist.get()
        else:
            crop_boxes = matched_detection_boxes
        # crops, true_sizes = self.crop_feature_map_keep_aspect_ratio(image, crop_boxes, [32, 256])
        path = os.path.join(self.flags.dump_directory, "metrics_input")
        return tf.py_func(self.write_to_file, [x, image, matched_detection_boxes, matched_transcriptions,
            filename, matched_detection_corpora, matched_groundtruth_text, path], x.dtype)

    def _build_metric_graph(self, words, detection_boxes, target_words, groundtruth_boxes, padded_groundtruth_text,
        debug_corpora, filename=None, original_image=None, string2int=None, int2string=None):
        """ Build a graph that computes Precision, Recall and Character Error Rate (CER). This graph is run once per eval
        stream.

        The first two metrics are coarse-grained, in that they measure accuracy at field level. For finer granularity we also
        compute CER, which is measured at character level. Moreover, Precision and Recall are more "accurate" metrics in the sense
        that they don't tollerate noise and in order to count a prediction as correct, all of its character have to match the groundtruth.

        Args:
            words: A list of size [self._top_paths] holding string Tensors of shape [num_detections]. The content of the tensor is
                a decoded string prediction.
            detection_boxes: A float32 Tensor of shape [num_detections, 4]. Detection boxes coming from stage 2 and CRNN pre-processing.
            target_words: A string Tensor of shape [num_detections]. The groundtruth text assigned to each detection box.
            groundtruth_boxes: A float32 Tensor of shape [groundtruth_size]. The groundtruth boxes from the input annotation.
            padded_groundtruth_text: A string Tensor of shape [padded_groundtruth_size]. The groundtruth text.
            debug_corpora: An int32 Tensor of shape [num_detections]. It contains the corpus of each detection box.
                This is unused for metrics computation, its only purpose is when self.flags.dump_metrics_input_to_tfrecord or
                self.flags.dump_metrics_input_to_tfrecord_using_groundtruth is on, in which case we want also to store
                the detection boxes corpus types in the output tfrecord.
            string2int: An optional HashTable object encoding strings to its internal alphabet. If None default to self._table_string2int.
            int2string: An optional HashTable object decoding codes from its internal alphabet to string. If None default to self._table_int2str.

        Returns:
            A float32 metric dictionary eval_metric_ops:
                'eval/precision',
                'eval/recall',
                'eval/CER'
        """
        if not string2int:
            string2int = self._table_str2int
        if not int2string:
            int2string = self._table_int2str

        with tf.name_scope('evaluation'):
            top_words = words[0]
            groundtruth_boxlist = box_list.BoxList(groundtruth_boxes)
            groundtruth_text = padded_groundtruth_text[:groundtruth_boxlist.num_boxes()]
            groundtruth_boxlist.add_field(fields.BoxListFields.groundtruth_text, groundtruth_text)
            detection_boxlist = box_list.BoxList(detection_boxes[0])
            detection_boxlist.add_field(fields.BoxListFields.transcription, top_words)
            dump_fn = None
            if self.flags.dump_metrics_input_to_tfrecord or self.flags.dump_metrics_input_to_tfrecord_using_groundtruth:
                dump_fn = partial(self.dump_tfrecord, debug_corpora=debug_corpora, detection_boxlist=detection_boxlist,
                    filename=filename, image=original_image)

            precision, precision_op = self.compute_precision(top_words, target_words, string2int, int2string)
            recall, recall_op, match =  self.compute_recall(groundtruth_boxlist, string2int, int2string, detection_boxlist=detection_boxlist)
            CER, CER_op, match = self.compute_CER(groundtruth_boxlist, string2int, match=match, dump_fn=dump_fn)

            eval_metric_ops = {
                'eval/precision' : (precision, precision_op),
                'eval/recall' : (recall, recall_op),
                'eval/CER' : (CER, CER_op)
            }
            return eval_metric_ops

    def _lstm_layers(self, field_features, detection_corpora, sequence_lengths):
        """ Implementation of bidirectional lstm. Also performs post-processing decoding of predictions.

        Args:
            field_features: A float32 Tensor of shape [num_detections, self._crop_size[1], self._crop_size[0] x D].
            detection_corpora: An int32 Tensor of shape [num_detections].
            sequence_lengths: An int64 Tensor of shape [num_detections]. The true width of the unpadded field_features.

        Returns:
            transcriptions_dict. Please refer to predict() for info on its structure.
        """
        parameters = self._parameters

        logprob, raw_pred = deep_bidirectional_lstm(field_features, detection_corpora, params=parameters, summaries=False)
        transcriptions_dict = {'prob': logprob,
                            'raw_predictions': raw_pred,
                            'sequence_lengths': sequence_lengths
                            }

        with tf.name_scope('code2str_conversion'):
            table_int2str = self._table_int2str
            sparse_code_pred, log_probability = tf.nn.ctc_beam_search_decoder(transcriptions_dict['prob'],
                                                                              sequence_length=sequence_lengths,
                                                                              merge_repeated=False,
                                                                              beam_width=100,
                                                                              top_paths=parameters.nb_logprob)

            transcriptions_dict[fields.TranscriptionResultFields.score] = log_probability

            sequence_lengths_pred = [tf.bincount(tf.cast(sparse_code_pred[i].indices[:, 0], tf.int32),
                                                minlength=tf.shape(transcriptions_dict['prob'])[1]) for i in range(parameters.top_paths)]

            pred_chars = [table_int2str.lookup(sparse_code_pred[i]) for i in range(parameters.top_paths)]

            list_preds = [get_words_from_chars(pred_chars[i].values, sequence_lengths=sequence_lengths_pred[i])
                          for i in range(parameters.top_paths)]

            transcriptions_dict[fields.TranscriptionResultFields.words] = tf.stack(list_preds)

        return transcriptions_dict

