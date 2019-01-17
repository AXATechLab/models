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
"""

Two-streams evaluation:

In order to evaluate the SYNTH and REAL datasets, we need to separate the two from the
input pipeline (we can only have one eval spec, so the input pipeline has to combine the two
datasets into one). In order to do this, the input pipeline keeps track of whether the read
image is SYNTH (also referred to as source) or REAL (also called target). This information
is in input_features['is_source_metrics'].

The Tensorflow estimator executes the update op of every metric for every example and calls
the variable read at the end of the input stream. A naive approach would be to condition both
operations on input_features['is_source_metrics']. However, because of this conditioning, the variable read could
only return one value depending on the condition, since it is requested only once.
With this approach, it is not possible to retrieve the metric value. Note that
the variable read does need to be conditioned otherwise tensorflow forbids the variable read.

My solution mimics the implementation of coco metrics: the update op simply stores the parameters
of each example evaluation. We maintain two datastructures, one for SYNTH and one for EVAL.
At variable read, we can finally evaluate metrics and read them without any conditioning.
Metric evaluation occurs by creating two sessions, one for each stream.

self._metric_names is a list used to establish an arbitrary ordering of the metric keys.
 We need the ordering to decide which is the first metric to evaluate.
 All the other metrics will depend on the first one. The first metric carries out the
 evaluation of ALL the metrics and stores the result in self._metrics.
"""


class CRNN:

    """
        Get the alphabet hash tables. This function is called when initializing a session
        for either train or eval streams.

        Args:
            parameters: A Params object, the CRNN parameters
    """
    def _get_tables(self, parameters):
        keys = [c for c in parameters.alphabet.encode('latin1')]
        values = parameters.alphabet_codes
        table_str2int = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.int64,
                    value_dtype=tf.int64), 0)#-1)

        keys = tf.cast(parameters.alphabet_decoding_codes, tf.int64)
        values = [c for c in parameters.alphabet_decoding]
        table_int2str = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(keys, values), '?')
        return table_str2int, table_int2str


    def __init__(self, parameters, detection_model, target_assigner, template_assigner,
        crop_size, start_at_step, backprop_feature_map, backprop_detection):

        # The fixed size which all detections will be resized to
        self._crop_size = [int(d) for d in crop_size]

        # Enable the CRNN at this global step
        self._start_at_step = start_at_step

        self.parameters = parameters

        # Access the detection model parameters. TODO: rethink this dependency
        self.detection_model = detection_model

        # whether to let gradient flow into the feature map or the detection layers
        self._backprop_feature_map = backprop_feature_map
        self._backprop_detection = backprop_detection

        # Standard Target assigner used to match groundtruth to detections or viceversa
        self.target_assigner = target_assigner

        # The template assigner used to match detections and groundtruth to the template
        self.template_assigner = template_assigner

        # The custom alphabet encoder and decoder
        self.table_str2int, self.table_int2str = self._get_tables(parameters)

        # Return zero loss in predict and eval mode
        self.zero_loss = tf.constant(0, dtype=tf.float32)

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
        # However, if CRNN is disabled, there's no reason to postprocess the predictions.
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

    """
       Build a placeholder prediction of CRNN. It can be used to return a null prediction.

        Args:
            detections_dict: the postprocessed detections to be integrated in the placeholder result
        Returns:
            A placeholder comforming to a CRNN prediction as far as type goes and with the
            given detections in it. Structure: [loss, transcription_dict, eval_metric_ops]
    """
    def no_result_fn(self, detections_dict):
        return lambda : self.no_forward_pass(detections_dict) + [self.no_eval_op]

    def no_forward_pass(self, detections_dict):
        """
       Build a placeholder forward pass of CRNN.

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
        return [self.zero_loss, transcriptions_dict]


    def predict(self, prediction_dict, true_image_shapes, mode):
        """
       Build the CRNN computational graph.

        Args:
            prediction_dict: the detections coming from stage 2
            true_image_shapes: the shape of the input image
            mode: train, eval or predict
        Returns:
            A placeholder comforming to a CRNN prediction as far as type goes and with the
            given detections in it.
        """
        self.debug_root_variable_scope = tf.get_variable_scope()
        with tf.variable_scope('crnn'):
            predict_fn = partial(self._predict, mode=mode, prediction_dict=prediction_dict,
                true_image_shapes=true_image_shapes)
            if mode == tf.estimator.ModeKeys.TRAIN:
                global_step = tf.train.get_or_create_global_step()
                disabled = tf.less(global_step, self._start_at_step)
            else:
                disabled = tf.constant(False, dtype=tf.bool)
            return tf.cond(disabled, self.no_result_fn(self.no_postprocessing),
                predict_fn, name="StepCond")

    def _predict(self, prediction_dict, true_image_shapes, mode):
        detection_model = self.detection_model
        # postprocess and unpad detections
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
        # Placeholder detection corpora
        detections_dict[fields.DetectionResultFields.detection_corpora] = tf.constant([[0]], dtype=tf.int32)
        padded_matched_transcriptions = tf.constant('', dtype=tf.string)
        # Remove detection classes since text detection is not a multiclass problem
        detections_dict.pop(fields.DetectionResultFields.detection_classes)

        normalized_boxlist = box_list.BoxList(normalized_detection_boxes)

        # Fetch groundtruth
        if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN]:
            gt_boxlists, gt_classes, _, gt_weights, gt_transcriptions = detection_model._format_groundtruth_data(true_image_shapes,
                stage='transcription')
            normalized_gt_boxlist = box_list.BoxList(tf.placeholder(shape=(1, 4), dtype=tf.float32))
            normalize_gt = lambda: box_list_ops.to_normalized_coordinates(gt_boxlists[0],
                true_image_shapes[0, 0], true_image_shapes[0, 1]).get()
            # Guard for examples with no objects to detect (box_list_ops throws an exception)
            normalized_gt_boxlist.set(tf.cond(gt_boxlists[0].num_boxes() > 0, normalize_gt,
                lambda: gt_boxlists[0].get()))

            # Switch this on to train on groundtruth
            # if True:#mode == tf.estimator.ModeKeys.TRAIN:
            #     normalized_boxlist = normalized_gt_boxlist
            #     num_detections = normalized_gt_boxlist.num_boxes()


            # Switch this on to train on groundtruth and detections
            # if mode == tf.estimator.ModeKeys.TRAIN:
            #     normalized_boxlist = box_list_ops.concatenate([normalized_boxlist, normalized_gt_boxlist])
            #     num_detections = num_detections + normalized_gt_boxlist.num_boxes()


        # Template Assignment (boxes with IOU bigger than 0.05 to some template space are mapped to that space)
        template_boxlist = box_list.BoxList(detection_model.curr_template_boxes)
        (_, _, _, _, match) = self.template_assigner.assign(normalized_boxlist, template_boxlist)
        template_corpora = detection_model.curr_template_corpora
        padded_detection_corpora = match.gather_based_on_match(template_corpora, -1, -1)

        # The name of the tf.cond operation
        BATCH_COND = 'BatchCond'

        if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN]:
            detection_boxlist = box_list_ops.to_absolute_coordinates(normalized_boxlist,
                true_image_shapes[0, 0], true_image_shapes[0, 1])

            detection_boxlist.add_field(fields.BoxListFields.scores, detection_scores)
            detection_boxlist.add_field(fields.BoxListFields.corpus, padded_detection_corpora)

            (_, cls_weights, _, _, match) = self.target_assigner.assign(detection_boxlist,
                gt_boxlists[0], gt_classes[0],
                unmatched_class_label=tf.constant(
                [1] + detection_model._num_classes * [0], dtype=tf.float32),
                groundtruth_weights=gt_weights[0])

            padded_matched_transcriptions = match.gather_based_on_match(gt_transcriptions[0], self.NULL, self.NULL)

            detection_boxlist.add_field(fields.BoxListFields.groundtruth_transcription, padded_matched_transcriptions)

            positive_indicator = match.matched_column_indicator()
            # valid_indicator = tf.logical_and( # not needed since the boxes are unpadded
            #     tf.range(detection_boxlist.num_boxes()) < num_detections,
            #     cls_weights > 0
            # )
            valid_indicator = cls_weights > 0
            sampled_indices = detection_model._second_stage_sampler.subsample(
                valid_indicator,
                None,
                positive_indicator,
                stage="transcription")
            # sampled_indices = tf.Print(sampled_indices, [], message="CRNN step")

            def train_forward_pass():
                sampled_boxlist = box_list_ops.boolean_mask(detection_boxlist, sampled_indices)
                # sampled_padded_boxlist = box_list_ops.pad_or_clip_box_list(
                #   sampled_boxlist,
                #   num_boxes=self.batch_size)

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

                transcriptions_dict = self._predict_lstm(rpn_features_to_crop, normalized_detection_boxes, matched_transcriptions, #
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
                update_op = tf.cond(self.debug_features['is_source_metrics'],
                    lambda: tf.py_func(source_update_op, common_args, []),
                    lambda: tf.py_func(target_update_op, common_args, []))

                # This var does the actual metric evaluation and stores the result in self._metrics
                first_var = tf.py_func(self._first_value_op, [], tf.float32)
                eval_metric_ops = {self._metric_names[0]: (first_var, update_op)}


                with tf.control_dependencies([first_var]):
                    for metric in self._metric_names[1:]:
                        eval_metric_ops[metric] = (tf.py_func(lambda m: self._metrics[m.decode('latin1')], [metric], tf.float32),
                            update_op)


                # Original Metrics
                # eval_metric_ops = self.compute_eval_ops(predictions_dict, padded_matched_transcriptions, sampled_indices,
                #     normalized_gt_boxlist, gt_transcriptions[0], debug_corpora=padded_detection_corpora)

            return [loss, predictions_dict, eval_metric_ops]


        predict_fn = lambda : [self.zero_loss, self._predict_lstm(rpn_features_to_crop, normalized_detection_boxes,
                padded_matched_transcriptions, detection_scores, padded_detection_corpora, num_detections, mode),
                self.no_eval_op]
        return tf.cond(tf.constant(True, dtype=tf.bool), predict_fn,
            self.no_result_fn(detections_dict), name=BATCH_COND)

    def crop_feature_map(self, features_to_crop, bboxes):
      output_height, output_width = self._crop_size
      # features_to_crop = tf.Print(features_to_crop, [tf.shape(features_to_crop)], message="features_to_crop", summarize=9999)
      def _keep_aspect_ratio_crop_and_resize(args):
        bbox, crop_width = args
        # dbshape = tf.cast(tf.shape(features_to_crop), dtype=tf.float32)
        # bbox = tf.Print(bbox, [dbshape[1] * (bbox[2] - bbox[0]), dbshape[2] * (bbox[3] - bbox[1])], message="how many cells under crop", summarize=99999)
        fixed_height_crop = tf.image.crop_and_resize(features_to_crop,
          tf.expand_dims(bbox, axis=0), [0], [output_height, crop_width])
        padded_crop = tf.pad(fixed_height_crop[0],
          [[0, 0], [0, output_width - crop_width], [0, 0]], "CONSTANT")
        # padded_crop, _ =  padding_inputs_width(fixed_height_crop[0], self._crop_size, 1)
        return padded_crop

      # aspect_ratios = (bboxes[:, 3] - bboxes[:, 1]) / (bboxes[:, 2] - bboxes[:, 0])
      # crop_widths = tf.math.minimum(tf.cast(tf.round(aspect_ratios * output_height), tf.int32),
      #   output_width)
      num_feat_map_cells = tf.cast(tf.shape(features_to_crop)[2], dtype=tf.float32) * (bboxes[:, 3] - bboxes[:, 1])
      # num_feat_map_cells = tf.Print(num_feat_map_cells, [num_feat_map_cells, tf.cast(tf.shape(features_to_crop)[1], dtype=tf.float32) * (bboxes[:, 2] - bboxes[:, 0])], message="num_feat_map_cells", summarize=9999)
      crop_widths = tf.math.minimum(tf.cast(tf.round(2.0 * num_feat_map_cells), tf.int32),
        output_width)
      crop_widths = tf.math.maximum(crop_widths, 1)
      # crop_widths = tf.Print(crop_widths, [crop_widths], message="crop_widths", summarize=99999)

      return shape_utils.static_or_dynamic_map_fn(
              _keep_aspect_ratio_crop_and_resize,
              elems=[bboxes, crop_widths],
              dtype=tf.float32,
              parallel_iterations=self.detection_model._parallel_iterations), crop_widths

    def crop_feature_map_debug(self, img, bboxes, crop_size):
      output_height, output_width = crop_size
      # features_to_crop = tf.Print(features_to_crop, [tf.shape(features_to_crop)], message="features_to_crop", summarize=9999)
      def _keep_aspect_ratio_crop_and_resize(args):
        bbox, crop_width = args
        # dbshape = tf.cast(tf.shape(features_to_crop), dtype=tf.float32)
        # bbox = tf.Print(bbox, [dbshape[1] * (bbox[2] - bbox[0]), dbshape[2] * (bbox[3] - bbox[1])], message="how many cells under crop", summarize=99999)
        fixed_height_crop = tf.image.crop_and_resize(img,
          tf.expand_dims(bbox, axis=0), [0], [output_height, crop_width])
        padded_crop = tf.pad(fixed_height_crop[0],
          [[0, 0], [0, output_width - crop_width], [0, 0]], "CONSTANT")
        # padded_crop, _ =  padding_inputs_width(fixed_height_crop[0], self._crop_size, 1)
        return padded_crop

      aspect_ratios = (bboxes[:, 3] - bboxes[:, 1]) / (bboxes[:, 2] - bboxes[:, 0])
      crop_widths = tf.math.minimum(tf.cast(tf.round(aspect_ratios * output_height), tf.int32),
        output_width)
      # num_feat_map_cells = tf.cast(tf.shape(features_to_crop)[2], dtype=tf.float32) * (bboxes[:, 3] - bboxes[:, 1])
      # crop_widths = tf.math.minimum(tf.cast(tf.round(2.0 * num_feat_map_cells), tf.int32),
      #   output_width)
      crop_widths = tf.math.maximum(crop_widths, 1)
      # crop_widths = tf.Print(crop_widths, [crop_widths], message="crop_widths", summarize=99999)

      return shape_utils.static_or_dynamic_map_fn(
              _keep_aspect_ratio_crop_and_resize,
              elems=[bboxes, crop_widths],
              dtype=tf.float32,
              parallel_iterations=self.detection_model._parallel_iterations), crop_widths

    def _first_value_op(self):
        g = tf.Graph()
        with g.as_default():
            types = [tf.string, tf.float32, tf.string, tf.int64, tf.float32, tf.string, tf.int64]
            shapes = [None, (1, None, 4), None, None, (None, 4), None, None]
            names = ["arg_{}".format(i) for i in range(len(types))]
            placeholders = [tf.placeholder(tp, name=n, shape=sh) for tp, n, sh in zip(types, names, shapes)]
            s2i, i2s = self._get_tables(self.parameters)
            compute_eval_ops = partial(self.compute_eval_ops, string2int=s2i, int2string=i2s)
            metrics = compute_eval_ops(*placeholders)
            ops = [v[1] for v in metrics.values()]
            variables = {k: var[0] for k, var in metrics.items()}

        def run_session(sess, preds):
            sess.run([s2i.init, i2s.init])
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
             # sess.run(tf.global_variables_initializer())
            for args in preds:
                sess.run(ops, feed_dict={"arg_{}:0".format(i): args[i] for i in range(len(args))})
                # Enable this to fetch per-example metrics
                # metrics = sess.run(variables)
                # pprint(metrics)
                # stream_vars_valid = [v for v in tf.local_variables() if 'evaluation/' in v.name]
                # sess.run(tf.variables_initializer(stream_vars_valid))

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
        detection_scores, detection_corpora, num_detections, mode):
        detection_model = self.detection_model
        # detection_boxes = tf.Print(detection_boxes, [detection_boxes], summarize=9999, message="detection_boxes")
        if not self._backprop_detection:
            detection_boxes = tf.stop_gradient(detection_boxes)
        if not self._backprop_feature_map:
            rpn_features_to_crop = tf.stop_gradient(rpn_features_to_crop)

        flattened_detected_feature_maps, seq_len_inputs = self.crop_feature_map(rpn_features_to_crop,
            detection_boxes)    # [batch, height, width, features]
        # flattened_detected_feature_maps tf.Print(flattened_detected_feature_maps, [tf.shape(flattened_detected_feature_maps)], message="flattened_detected_feature_maps", summarize=99999)

        # Code to bypass crop and resize
        # orig_image = self.debug_features[fields.InputDataFields.image]
        # flattened_detected_feature_maps, seq_len_inputs = self.crop_feature_map_debug(orig_image, detection_boxes, [x * 16 for x in self._crop_size])
        # Debug bypass
        # seq_len_inputs = tf.py_func(self.write_to_file, [seq_len_inputs, flattened_detected_feature_maps, self.debug_features['debug'],
        #     tf.tile(tf.constant([-1], dtype=tf.int64), [tf.shape(flattened_detected_feature_maps)[0]]),
        #     tf.tile(tf.constant(['$']), [tf.shape(flattened_detected_feature_maps)[0]]),
        #     seq_len_inputs],
        #     seq_len_inputs.dtype)
        # - Debug bypass
        # with tf.variable_scope(self.debug_root_variable_scope, reuse=True):
        #     flattened_detected_feature_maps, self.endpoints = (
        #     detection_model._feature_extractor.extract_proposal_features(
        #         flattened_detected_feature_maps,
        #         scope=detection_model.first_stage_feature_extractor_scope))
        #     seq_len_inputs = tf.cast(seq_len_inputs / 16, dtype=seq_len_inputs.dtype)

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
        detections_dict[fields.DetectionResultFields.num_detections] = tf.cast(num_detections, dtype=tf.float32)
        for k,v in detections_dict.items():
            detections_dict[k] = tf.expand_dims(v, axis=0)
        transcription_dict.update(detections_dict)
        return transcription_dict


    def str2code(self, labels, table_str2int=None):
        # Convert string label to code label
        with tf.name_scope('str2code_conversion'):
            if not table_str2int:
             table_str2int = self.table_str2int
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
        # seq_len_inputs = tf.Print(seq_len_inputs, [seq_len_inputs], message="seq_len_inputs", summarize=99999)
        # Loss
        # ----
        # >>> Cannot have longer labels than predictions -> error
        with tf.control_dependencies([tf.less_equal(sparse_code_target.dense_shape[1], tf.reduce_max(tf.cast(seq_len_inputs, tf.int64)))]):
            loss_ctc = tf.nn.ctc_loss(labels=sparse_code_target,
                                      inputs=predictions_dict['prob'],
                                      sequence_length=seq_len_inputs,
                                      preprocess_collapse_repeated=False,
                                      ctc_merge_repeated=True,
                                      ignore_longer_outputs_than_inputs=True,  # returns zero gradient in case it happens -> ema loss = NaN
                                      time_major=True)
            # loss_ctc = tf.cond(tf.is_nan(loss_ctc), lambda: tf.Print(0, [], message="NaN loss"),
            #     lambda: tf.reduce_mean(loss_ctc))
            loss_ctc = tf.reduce_mean(loss_ctc)

            seq_lengths_labels = tf.bincount(tf.cast(sparse_code_target.indices[:, 0], tf.int32), #array of labels length
                                         minlength= tf.shape(predictions_dict['prob'])[1])
            # loss_ctc = tf.Print(loss_ctc, [loss_ctc, seq_lengths_labels], message='* Loss : ', summarize=10000000)
        return loss_ctc


    def write_to_file(self, dummy, dt_crops, source_id, corpora, transcriptions, true_sizes):
        # np.savetxt('/notebooks/Detection/workspace/models/E2E/cer_48/zoom_in/tf1.12/predictions/{}.txt'.format(source_id),
        #     np.concatenate([dt, gt], axis=1))
        path = '/notebooks/Detection/workspace/models/E2E/cer_48/zoom_in/tf1.12/predictions/{}.tfrecord'.format(source_id)
        print("!!! The hard coded path is", path)
        debug_writer = tf.python_io.TFRecordWriter(path)
        for i in range(dt_crops.shape[0]):
            crop = dt_crops[i]
            # du.write_tfrecord_example(debug_writer, crop, "{}_crop{}".format(features['debug'], i), "None",
            #     crop[0], crop[1], np.array([0]), transcriptions[i], False, False, 0)
            du.write_old_tfrecord_example(debug_writer, crop[:, :true_sizes[i]], corpora[i], transcriptions[i])
        debug_writer.close()
        return dummy

    def compute_eval_ops(self, words, detection_boxes, matched_transcriptions, dt_positive_indices, gt_boxes, gt_transcriptions,
        debug_corpora=None, string2int=None, int2string=None):
        """
            TODO: Update comments
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
            string2int = self.table_str2int
        if not int2string:
            int2string = self.table_int2str

        with tf.name_scope('evaluation'):
            def encode_groundtruth(matched_transcriptions):
                matched_codes = self.str2code(matched_transcriptions, string2int)
                seq_lengths_labels = tf.bincount(tf.cast(matched_codes.indices[:, 0], tf.int32), #array of labels length
                                 minlength= tf.shape(matched_transcriptions)[0])
                target_chars = int2string.lookup(tf.cast(matched_codes, tf.int64))
                return get_words_from_chars(target_chars.values, seq_lengths_labels)
            # Debug functions
            def print_string_tensor(source, dest, mess, summar=9999):
                tens = tf.concat([tf.expand_dims(x, axis=1) for x in [source, dest]], axis=1)
                return tf.map_fn(lambda t: tf.Print(t[0],[*tf.split(t, 2, axis=0), tf.shape(tens)[0]], message=mess, summarize=summar), tens)

            gt_boxlist = box_list.BoxList(gt_boxes)
            all_predictions = words[0]

            # Compute Precision
            target_words = encode_groundtruth(matched_transcriptions)
            # all_predictions = print_string_tensor(all_predictions, target_words, "precision")
            precision, precision_op = tf.metrics.accuracy(target_words, all_predictions,
                name='precision')


            # Compute Recall
            detection_boxlist = box_list.BoxList(detection_boxes[0])
            (_, _, _, _, match) = self.target_assigner.assign(gt_boxlist, detection_boxlist)
            padded_best_predictions = match.gather_based_on_match(all_predictions, self.NULL, self.NULL)

            unpadded_gt_transcriptions = gt_transcriptions[:gt_boxlist.num_boxes()]
            target_words = encode_groundtruth(unpadded_gt_transcriptions)
            # target_words = print_string_tensor(target_words, padded_best_predictions, "recall")
            recall, recall_op = tf.metrics.accuracy(target_words, padded_best_predictions,
                name='recall')

            # Compute Character Error Rate
            indicator = match.matched_column_indicator()
            # Enable this to select one type
            # indicator = tf.logical_and(match.matched_column_indicator(), )
            sampled_matched_transcriptions = tf.boolean_mask(unpadded_gt_transcriptions, indicator)
            sampled_matched_predictions = tf.boolean_mask(padded_best_predictions, indicator)

            # Compare with old arch
            # padded_dets_boxes = match.gather_based_on_match(detection_boxlist.get(), [-1.0] * 4, [-1.0] * 4)
            # padded_dets_corpora = match.gather_based_on_match(debug_corpora, -2, -2)
            # best_dets_boxes = tf.boolean_mask(padded_dets_boxes, indicator)
            # matched_gt_boxes = tf.boolean_mask(gt_boxlist.get(), indicator)
            # dets_corpora = tf.boolean_mask(padded_dets_corpora, indicator)
            # img = self.debug_features[fields.InputDataFields.image]
            # # Enable this line to bypass detection and crop using groundtruth
            ## best_dets_boxes = matched_gt_boxes
            # crops, true_sizes = self.crop_feature_map_debug(img, best_dets_boxes, [32, 256])
            # sampled_matched_predictions = tf.py_func(self.write_to_file, [sampled_matched_predictions, crops, self.debug_features['debug'], dets_corpora, sampled_matched_transcriptions, true_sizes],
            #         sampled_matched_predictions.dtype)

            # Compute CER on non-empty vectors
            sampled_matched_transcriptions = tf.cond(tf.shape(sampled_matched_transcriptions)[0] < 1,
                lambda: tf.constant([self.NULL], dtype=tf.string),
                lambda: sampled_matched_transcriptions)
            sampled_matched_predictions = tf.cond(tf.shape(sampled_matched_predictions)[0] < 1,
                lambda: tf.constant([self.NULL], dtype=tf.string),
                lambda: sampled_matched_predictions)
            # sampled_matched_transcriptions = tf.Print(sampled_matched_transcriptions, [tf.shape(sampled_matched_transcriptions)], message="sampled_matched_transcriptions", summarize=9999)
            # sampled_matched_predictions = tf.Print(sampled_matched_predictions, [tf.shape(sampled_matched_predictions)], message="sampled_matched_predictions", summarize=9999)

            # sampled_matched_transcriptions = tf.boolean_mask(matched_transcriptions, dt_positive_indices)
            # sampled_matched_predictions = tf.boolean_mask(all_predictions, dt_positive_indices)
            sparse_code_target = self.str2code(sampled_matched_transcriptions, string2int)
            sparse_code_pred = self.str2code(sampled_matched_predictions, string2int)
            db0 = [sparse_code_pred.indices, sparse_code_pred.values, sparse_code_pred.dense_shape]
            db = [sparse_code_target.indices, sparse_code_target.values, sparse_code_target.dense_shape]
            # sparse_code_target = tf.SparseTensor(db[0], db[1],
            #     tf.Print(db[2], db + db0 + [sampled_matched_predictions], message="sparse_code_target", summarize=99999))
            CER, CER_op = tf.metrics.mean(tf.edit_distance(tf.cast(sparse_code_pred, dtype=tf.int64),
                 tf.cast(sparse_code_target, dtype=tf.int64)), name='CER')

            # inits = [precision.initializer, recall.initializer, CER.initializer]

            # Print to console
            # precision = tf.Print(precision, [precision], message="Precision -- ", name='precision')
            # recall = tf.Print(recall, [recall], message="Recall -- ", name="recall")
            # CER = tf.Print(CER, [CER], message="CER -- ", name='CER')

            # CER_op = tf.Print(CER_op, [predictions_dict['words'][0]], summarize=100)
            eval_metric_ops = {
                'eval/precision' : (precision, precision_op),
                'eval/recall' : (recall, recall_op),
                'eval/CER' : (CER, CER_op)
            }
            return eval_metric_ops

    # Code from crnn_fn
    def lstm_layers(self, feature_maps, corpus, seq_len_inputs, mode):
        parameters = self.parameters

        # seq_len_inputs = tf.Print(seq_len_inputs, [seq_len_inputs], message="seq_len_inputs", summarize=99999)


        logprob, raw_pred = deep_bidirectional_lstm(feature_maps, corpus, params=parameters, summaries=False)
        # seq_len_inputs = tf.zeros_like(features['corpus']) + features['image_width']
        predictions_dict = {'prob': logprob,
                            'raw_predictions': raw_pred,
                            'seq_len_inputs': seq_len_inputs
                            }

        with tf.name_scope('code2str_conversion'):
            table_int2str = self.table_int2str
            # seq_len_inputs = tf.Print(seq_len_inputs, [seq_len_inputs], message="seq_len_inputs", summarize=9999)
            sparse_code_pred, log_probability = tf.nn.ctc_beam_search_decoder(predictions_dict['prob'],
                                                                              sequence_length=seq_len_inputs,
                                                                              merge_repeated=False,
                                                                              beam_width=100,
                                                                              top_paths=parameters.nb_logprob)
            # confidence value

            predictions_dict[fields.TranscriptionResultFields.score] = log_probability

            sequence_lengths_pred = [tf.bincount(tf.cast(sparse_code_pred[i].indices[:, 0], tf.int32),
                                                minlength=tf.shape(predictions_dict['prob'])[1]) for i in range(parameters.top_paths)]

            pred_chars = [table_int2str.lookup(sparse_code_pred[i]) for i in range(parameters.top_paths)]

            list_preds = [get_words_from_chars(pred_chars[i].values, sequence_lengths=sequence_lengths_pred[i])
                          for i in range(parameters.top_paths)]

            predictions_dict[fields.TranscriptionResultFields.words] = tf.stack(list_preds)
            # predictions_dict['words'] = tf.Print(predictions_dict['words'], [predictions_dict['words'][0]], message="predictions_dict['words']", summarize=100)

        return predictions_dict

