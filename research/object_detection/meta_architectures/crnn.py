import tensorflow as tf

from object_detection.core import box_list, box_list_ops
from object_detection.core import standard_fields as fields
import sys
sys.path.append("/notebooks/Transcription/tf-crnn")
from tf_crnn.model import deep_bidirectional_lstm, get_words_from_chars
from tf_crnn.config import  CONST
from functools import partial
from utils import shape_utils

class CRNN:

    def __init__(self, parameters, detection_model, target_assigner, template_assigner,
        crop_size, start_at_step, backprop_feature_map, backprop_detection):
        self._crop_size = [int(d) for d in crop_size]
        self._start_at_step = start_at_step
        self.parameters = parameters
        self.detection_model = detection_model
        self._backprop_feature_map = backprop_feature_map
        self._backprop_detection = backprop_detection
        self.target_assigner = target_assigner
        self.template_assigner = template_assigner
        keys = [c for c in parameters.alphabet.encode('latin1')]
        values = parameters.alphabet_codes
        self.table_str2int = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.int64,
                    value_dtype=tf.int64), -1)
        keys = tf.cast(parameters.alphabet_decoding_codes, tf.int64)
        values = [c for c in parameters.alphabet_decoding]
        self.table_int2str = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(keys, values), '?')


        self.zero_loss = tf.constant(0, dtype=tf.float32)

        self.no_eval_op = {
            'eval/precision' : (tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32)),
            'eval/recall' : (tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32)),
            'eval/CER' : (tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32))
        }
        self.no_postprocessing = {
            'detection_boxes' : tf.constant(0, dtype=tf.float32),
            'detection_scores' : tf.constant(0, dtype=tf.float32),
            'detection_corpora' : tf.constant(0, dtype=tf.int32),
            'num_detections' : tf.constant(0, dtype=tf.float32)
        }

    def no_result_fn(self, detections_dict):
        return lambda : self.no_forward_pass(detections_dict) + [self.no_eval_op]

    def no_forward_pass(self, detections_dict):
        transcriptions_dict = {
            'raw_predictions': tf.constant(0, dtype=tf.int64),
            'labels': tf.constant('', dtype=tf.string),
            'seq_len_inputs': 0,
            'prob': tf.constant([[0], [0]], dtype=tf.float32),
            'score': tf.constant(0, dtype=tf.float32),
            'words': tf.constant('', dtype=tf.string)
        }
        transcriptions_dict.update(detections_dict)
        return [self.zero_loss, transcriptions_dict]

    # This is in the same fashion as predict_third_stage's inference
    # TODO: use parameters instead of copying FasterRCNN's
    def predict(self, prediction_dict, true_image_shapes, mode):
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
        # Postprocess FasterRCNN stage 2
        detection_model = self.detection_model
        detections_dict = detection_model._postprocess_box_classifier(
            prediction_dict['refined_box_encodings'],
            prediction_dict['class_predictions_with_background'],
            prediction_dict['proposal_boxes'],
            prediction_dict['num_proposals'],
            prediction_dict['proposal_corpora'],
            true_image_shapes)
        num_detections = tf.cast(detections_dict[
            fields.DetectionResultFields.num_detections][0], tf.int32)
        normalized_detection_boxes = detections_dict[
          fields.DetectionResultFields.detection_boxes][0][:num_detections]

        detection_scores = detections_dict[
            fields.DetectionResultFields.detection_scores][0][:num_detections]
        # detection_corpora = detections_dict[
        #     fields.DetectionResultFields.detection_corpora][0]
        padded_matched_transcriptions = tf.constant('', dtype=tf.string)
        detections_dict.pop('detection_classes')
        # num_detections = tf.Print(num_detections, [num_detections], message="Num detections")
        rpn_features_to_crop = prediction_dict['rpn_features_to_crop']

        # Corpora assignment
        normalized_boxlist = box_list.BoxList(normalized_detection_boxes)

        if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN]:
            gt_boxlists, gt_classes, _, gt_weights, gt_transcriptions = detection_model._format_groundtruth_data(true_image_shapes,
                stage='transcription')

        # Switch this on to train on groundtruth
        # if mode == tf.estimator.ModeKeys.TRAIN:
        #     normalized_boxlist = box_list_ops.to_normalized_coordinates(gt_boxlists[0],
        #         true_image_shapes[0, 0], true_image_shapes[0, 1])


        template_boxlist = box_list.BoxList(tf.constant(detection_model.template_proposals, dtype=tf.float32))
        (_, _, _, _, match) = self.template_assigner.assign(normalized_boxlist, template_boxlist)
        template_corpora = tf.constant(detection_model.template_corpora, dtype=tf.int32)
        padded_detection_corpora = match.gather_based_on_match(template_corpora, -1, -1)
        # Filter out false positives, TODO: move in EVAL
        # positive_indicator = match.matched_column_indicator()
        # valid_indicator = tf.range(detection_boxlist.num_boxes()) < num_detections
        # sampled_indices = tf.logical_and(positive_indicator, valid_indicator)
        # detection_boxlist = box_list_ops.boolean_mask(detection_boxlist, sampled_indices)

        BATCH_COND = 'BatchCond'
        NULL = '?' # Question marks are unmapped so they will never be matched
        # rpn_features_to_crop = tf.Print(rpn_features_to_crop, [tf.shape(rpn_features_to_crop)], message="The size of the Feature Map is", summarize=9999)

        if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN]:
            # gt_boxlists, gt_classes, _, gt_weights, gt_transcriptions = detection_model._format_groundtruth_data(true_image_shapes,
            #     stage='transcription')

            # gt_transcriptions = tf.Print(gt_transcriptions, [gt_transcriptions, tf.shape(gt_transcriptions)], message="CRNN received this transcr.", summarize=99999)

            detection_boxlist = box_list_ops.to_absolute_coordinates(normalized_boxlist,
                true_image_shapes[0, 0], true_image_shapes[0, 1])

            detection_boxlist.add_field(fields.BoxListFields.scores, detection_scores)
            detection_boxlist.add_field(fields.BoxListFields.corpus, padded_detection_corpora)

            (_, cls_weights, _, _, match) = self.target_assigner.assign(detection_boxlist,
                gt_boxlists[0], gt_classes[0],
                unmatched_class_label=tf.constant(
                [1] + detection_model._num_classes * [0], dtype=tf.float32),
                groundtruth_weights=gt_weights[0])

            padded_matched_transcriptions = match.gather_based_on_match(gt_transcriptions[0], NULL, NULL) # This list is padded with NULL
            # detection_transcriptions = tf.Print(detection_transcriptions, [detection_transcriptions], message="These are the matched GTs transcr.", summarize=99999)
            detection_boxlist.add_field(fields.BoxListFields.groundtruth_transcription, padded_matched_transcriptions)

            positive_indicator = match.matched_column_indicator()
            # positive_indicator = tf.Print(positive_indicator, [match.matched_column_indices()], summarize=1000, message="Indices")
            # positive_indicator = tf.Print(positive_indicator, [gt_transcriptions[0]], summarize=1000, message="Num GTs")

            # positive_indicator = tf.Print(positive_indicator, [positive_indicator], message="positive_indicator", summarize=99999)
            valid_indicator = tf.logical_and(
                tf.range(detection_boxlist.num_boxes()) < num_detections, # not needed
                cls_weights > 0
            )
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
                # matched_transcriptions = tf.Print(matched_transcriptions, [matched_transcriptions], message="matched_transcriptions", summarize=1000)
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
                loss, predictions_dict = tf.cond(tf.equal(tf.shape(sampled_indices)[0], 0),
                    lambda : self.no_forward_pass(detections_dict), train_forward_pass, name=BATCH_COND)
                eval_metric_ops = self.no_eval_op
            else:
                loss, predictions_dict = tf.cond(tf.constant(False, dtype=tf.bool),
                    lambda : self.no_forward_pass(detections_dict), eval_forward_pass, name=BATCH_COND)
                eval_metric_ops = self.compute_eval_ops(predictions_dict, padded_matched_transcriptions, sampled_indices,
                    gt_transcriptions[0])
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
        # crop_width = tf.Print(crop_width, [crop_width], message="Crop Width", summarize=9999)
        padded_crop = tf.pad(fixed_height_crop[0],
          [[0, 0], [0, output_width - crop_width], [0, 0]], "CONSTANT")
        return padded_crop

      aspect_ratios = (bboxes[:, 3] - bboxes[:, 1]) / (bboxes[:, 2] - bboxes[:, 0])
      crop_widths = tf.math.minimum(tf.cast(tf.round(aspect_ratios * output_height), tf.int32),
        output_width)

      # bboxes = tf.Print(bboxes, [tf.shape(bboxes), tf.shape(crop_widths), tf.shape(aspect_ratios)], message="bboxes", summarize=9999)
      return shape_utils.static_or_dynamic_map_fn(
              _keep_aspect_ratio_crop_and_resize,
              elems=[bboxes, crop_widths],
              dtype=tf.float32,
              parallel_iterations=self.detection_model._parallel_iterations), crop_widths


    # def keep_aspect_ratio_crop_and_resize(self, rpn_features_to_crop, detection_boxes):
    #     normalized_crop_width = self._crop_size[1] / tf.shape(rpn_features_to_crop)[2]
    #     ymin, xmin, ymax, xmax = tf.split(detection_boxes, 4, axis=1)
    #     fixed_xmax = xmin + normalized_crop_width
    #     fixed_detection_boxes = tf.concat([ymin, xmin, ymax, fixed_xmax], axis=1)
    #     wider_crops = flattened_detected_feature_maps = (
    #           self.detection_model._compute_second_stage_input_feature_maps(
    #               rpn_features_to_crop, tf.expand_dims(fixed_detection_boxes, axis=0), stage='transcription',
    #               crop_size=self._crop_size)
    #           )


    def _predict_lstm(self, rpn_features_to_crop, detection_boxes, matched_transcriptions,
        detection_scores, detection_corpora, num_detections, mode):
        # Reuse the second stage cropping as-is
        detection_model = self.detection_model
        if not self._backprop_detection:
            detection_boxes = tf.stop_gradient(detection_boxes)
        if not self._backprop_feature_map:
            rpn_features_to_crop = tf.stop_gradient(rpn_features_to_crop)

        flattened_detected_feature_maps, seq_len_inputs = self.crop_feature_map(rpn_features_to_crop,
            detection_boxes)    # [batch, height, width, features]


        with tf.variable_scope('Reshaping_cnn'):
            n_channels = flattened_detected_feature_maps.get_shape().as_list()[3]
            transposed = tf.transpose(flattened_detected_feature_maps, perm=[0, 2, 1, 3],
                                      name='transposed')  # [batch, width, height, features]
            conv_reshaped = tf.reshape(transposed, [-1, self._crop_size[1], self._crop_size[0] * n_channels],
                                       name='reshaped')  # [batch, width, height x features]
        # detection_corpora = tf.Print(detection_corpora, [detection_corpora], message="corpora", summarize=100000)
        transcription_dict = self.lstm_layers(conv_reshaped, detection_corpora, seq_len_inputs, mode)
        transcription_dict['labels'] = matched_transcriptions
        # transcription_dict['label_codes'] = sparse_code_target
        detections_dict = {}
        detections_dict['detection_boxes'] = detection_boxes
        detections_dict['detection_scores'] = detection_scores
        detections_dict['detection_corpora'] = detection_corpora
        detections_dict['num_detections'] = tf.cast(num_detections, dtype=tf.float32)
        for k,v in detections_dict.items():
            detections_dict[k] = tf.expand_dims(v, axis=0)
        transcription_dict.update(detections_dict)
        return transcription_dict


    def str2code(self, labels):
        # Convert string label to code label
        with tf.name_scope('str2code_conversion'):
            table_str2int = self.table_str2int
            splitted = tf.string_split(labels, delimiter='')
            values_int = tf.cast(tf.squeeze(tf.decode_raw(splitted.values, tf.uint8)), tf.int64)
            # values_int = tf.Print(values_int, [labels], summarize=9999999)
            codes = table_str2int.lookup(values_int)
            codes = tf.cast(codes, tf.int32)
            # codes = tf.Print(codes, [codes, labels], summarize=1000)
            return tf.SparseTensor(splitted.indices, codes, splitted.dense_shape)

    def loss(self, predictions_dict, sparse_code_target):
        # Alphabet and codes
        seq_len_inputs = predictions_dict['seq_len_inputs']
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


    def compute_eval_ops(self, predictions_dict, matched_transcriptions, sampled_indices, gt_transcriptions):
        """
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
        with tf.name_scope('evaluation'):
            def encode_groundtruth(matched_transcriptions):
                matched_codes = self.str2code(matched_transcriptions)
                seq_lengths_labels = tf.bincount(tf.cast(matched_codes.indices[:, 0], tf.int32), #array of labels length
                                 minlength= tf.shape(matched_transcriptions)[0])
                target_chars = self.table_int2str.lookup(tf.cast(matched_codes, tf.int64))
                return get_words_from_chars(target_chars.values, seq_lengths_labels), matched_codes

            # Compute Precision
            target_words, _ = encode_groundtruth(matched_transcriptions)
            precision, precision_op = tf.metrics.accuracy(target_words, predictions_dict['words'][0],
                name='precision')

            # Compute Recall
            sampled_matched_transcriptions = tf.boolean_mask(matched_transcriptions, sampled_indices)
            num_groundtruths = gt_transcriptions.get_shape().as_list()[0]
            num_matches = tf.shape(sampled_matched_transcriptions)[0]
            pad_size = [[0, num_groundtruths - num_matches]]
            target_words, sparse_code_target = encode_groundtruth(sampled_matched_transcriptions)
            # target_words = tf.Print(target_words, [tf.shape(target_words)], summarize=10000, message="Number of eval assignments")
            padded_target_words = tf.pad(target_words,
                paddings=pad_size, constant_values='groundtruth')
            sampled_matched_predictions = tf.boolean_mask(predictions_dict['words'][0], sampled_indices)
            # sampled_matched_predictions = tf.Print(sampled_matched_predictions, [tf.shape(sampled_matched_predictions)], summarize=10000, message="shape of pred words")

            padded_matched_predictions = tf.pad(sampled_matched_predictions,
                paddings=pad_size, constant_values='prediction')
            recall, recall_op = tf.metrics.accuracy(padded_target_words, padded_matched_predictions,
                name='recall')

            # Compute Character Error Rate
            sparse_code_pred = self.str2code(sampled_matched_predictions)
            CER, CER_op = tf.metrics.mean(tf.edit_distance(tf.cast(sparse_code_pred, dtype=tf.int64),
                 tf.cast(sparse_code_target, dtype=tf.int64)), name='CER')

            # Print to console
            precision = tf.Print(precision, [precision], message="Precision -- ")
            recall = tf.Print(recall, [recall], message="Recall -- ")
            CER = tf.Print(CER, [CER], message="CER -- ")
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

            sparse_code_pred, log_probability = tf.nn.ctc_beam_search_decoder(predictions_dict['prob'],
                                                                              sequence_length=seq_len_inputs,
                                                                              merge_repeated=False,
                                                                              beam_width=100,
                                                                              top_paths=parameters.nb_logprob)
            # confidence value

            predictions_dict['score'] = log_probability

            sequence_lengths_pred = [tf.bincount(tf.cast(sparse_code_pred[i].indices[:, 0], tf.int32),
                                                minlength=tf.shape(predictions_dict['prob'])[1]) for i in range(parameters.top_paths)]

            pred_chars = [table_int2str.lookup(sparse_code_pred[i]) for i in range(parameters.top_paths)]

            list_preds = [get_words_from_chars(pred_chars[i].values, sequence_lengths=sequence_lengths_pred[i])
                          for i in range(parameters.top_paths)]

            predictions_dict['words'] = tf.stack(list_preds)
            # predictions_dict['words'] = tf.Print(predictions_dict['words'], [predictions_dict['words'][0]], message="predictions_dict['words']", summarize=100)

        return predictions_dict

