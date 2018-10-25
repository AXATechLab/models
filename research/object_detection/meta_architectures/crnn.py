import tensorflow as tf

from core import box_list, box_list_ops
from core import standard_fields as fields
import sys
sys.path.append("/notebooks/Transcription/tf-crnn")
from src.model import deep_bidirectional_lstm, get_words_from_chars
from src.config import  CONST
from functools import partial

class CRNN:

    def __init__(self, parameters, detection_model, target_assigner, is_training):
        self.parameters = parameters
        self.detection_model = detection_model
        self.target_assigner = target_assigner
        keys = [c for c in parameters.alphabet.encode('latin1')]
        values = parameters.alphabet_codes
        self.table_str2int = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.int64, value_dtype=tf.int64), -1)
        keys = tf.cast(parameters.alphabet_decoding_codes, tf.int64)
        values = [c for c in parameters.alphabet_decoding]
        self.table_int2str = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(keys, values), '?')


        self.zero_loss = tf.constant(0, dtype=tf.float32)

        self.no_eval_op = {
            'eval/precision' : (tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32)),
            'eval/recall' : (tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32))
        }
        self.no_postprocessing = {
            'detection_boxes' : tf.constant(0, dtype=tf.float32),
            'detection_scores' : tf.constant(0, dtype=tf.float32),
            'num_detections' : tf.constant(0, dtype=tf.float32)
        }

    def no_result_fn(self, detections_dict):
        return lambda : self.no_forward_pass(detections_dict) + [self.no_eval_op]

    def no_forward_pass(self, detections_dict):
        transcriptions_dict = {
            'raw_predictions': tf.constant(0, dtype=tf.int64),
            'labels': tf.constant('', dtype=tf.string), 
            # 'label_codes': tf.constant('', dtype=tf.string), 
            'seq_len_inputs': 0,
            'prob': tf.constant([[0], [0]], dtype=tf.float32),
            'score': tf.constant(0, dtype=tf.float32),
            'words': tf.constant('?', dtype=tf.string)
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
                disabled = tf.less(global_step, 1)
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
            true_image_shapes)
        detection_boxes = detections_dict[
          fields.DetectionResultFields.detection_boxes][0]
        detection_scores = detections_dict[
            fields.DetectionResultFields.detection_scores][0]
        matched_transcriptions = tf.constant('', dtype=tf.string)
        detections_dict.pop('detection_classes')

        num_detections = tf.cast(detections_dict[
            fields.DetectionResultFields.num_detections][0], tf.int32)
        rpn_features_to_crop = prediction_dict['rpn_features_to_crop']
        BATCH_COND = 'BatchCond'
        NULL = '?' # Question marks are unmapped so they will never be matched
        # rpn_features_to_crop = tf.Print(rpn_features_to_crop, [tf.shape(rpn_features_to_crop)], message="The size of the Feature Map is", summarize=9999)

        if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN]:
            gt_boxlists, gt_classes, _, gt_weights, gt_transcriptions = detection_model._format_groundtruth_data(true_image_shapes, 
                stage='transcription')

            # gt_transcriptions = tf.Print(gt_transcriptions, [gt_transcriptions, tf.shape(gt_transcriptions)], message="CRNN received this transcr.", summarize=99999)

            detection_boxlist = box_list_ops.to_absolute_coordinates(box_list.BoxList(detection_boxes),
                true_image_shapes[0, 0], true_image_shapes[0, 1])
            detection_boxlist.add_field(fields.BoxListFields.scores, detection_scores)

            (_, cls_weights, _, _, match) = self.target_assigner.assign(detection_boxlist, 
                gt_boxlists[0], gt_classes[0],         
                unmatched_class_label=tf.constant(
                [1] + detection_model._num_classes * [0], dtype=tf.float32),
                groundtruth_weights=gt_weights[0])

            matched_transcriptions = match.gather_based_on_match(gt_transcriptions[0], NULL, NULL)
            # detection_transcriptions = tf.Print(detection_transcriptions, [detection_transcriptions], message="These are the matched GTs transcr.", summarize=99999)
            detection_boxlist.add_field(fields.BoxListFields.groundtruth_transcription, matched_transcriptions)


            positive_indicator = match.matched_column_indicator()
            # positive_indicator = tf.Print(positive_indicator, [positive_indicator], message="positive_indicator", summarize=99999)
            valid_indicator = tf.logical_and(
                tf.range(detection_boxlist.num_boxes()) < num_detections,
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
                detection_boxes = sampled_padded_boxlist.get()
                matched_transcriptions = sampled_padded_boxlist.get_field(fields.BoxListFields.groundtruth_transcription)
                # detection_transcriptions = tf.Print(detection_transcriptions, [detection_transcriptions], message="These are the subsampled GTs transcr.", summarize=99999)
                detection_scores = sampled_padded_boxlist.get_field(fields.BoxListFields.scores)
                # tf.minimum(sampled_boxlist.num_boxes(), self.batch_size)
                num_detections = sampled_boxlist.num_boxes()           
                sparse_code_target = self.str2code(matched_transcriptions)

                transcriptions_dict = self._predict_lstm(rpn_features_to_crop, detection_boxes, matched_transcriptions, 
                        detection_scores, num_detections, mode)

                return [self.loss(transcriptions_dict, sparse_code_target), transcriptions_dict]
            def eval_forward_pass():
                transcriptions_dict = self._predict_lstm(rpn_features_to_crop, detection_boxes, matched_transcriptions, 
                        detection_scores, num_detections, mode)
                return [self.zero_loss, transcriptions_dict]

            if mode == tf.estimator.ModeKeys.TRAIN:
                loss, predictions_dict = tf.cond(tf.equal(tf.shape(sampled_indices)[0], 0), 
                    lambda : self.no_forward_pass(detections_dict), train_forward_pass, name=BATCH_COND)   
                eval_metric_ops = self.no_eval_op
            else:
                loss, predictions_dict = tf.cond(tf.constant(False, dtype=tf.bool), 
                    lambda : self.no_forward_pass(detections_dict), eval_forward_pass, name=BATCH_COND)   
                eval_metric_ops = self.compute_eval_ops(predictions_dict, detection_boxlist, sampled_indices, 
                    gt_transcriptions[0])
            return [loss, predictions_dict, eval_metric_ops]


        predict_fn = lambda : [self.zero_loss, self._predict_lstm(rpn_features_to_crop, detection_boxes, 
                matched_transcriptions, detection_scores, num_detections, mode), self.no_eval_op]
        return tf.cond(tf.constant(True, dtype=tf.bool), predict_fn, 
            self.no_result_fn(detections_dict), name=BATCH_COND)



    def _predict_lstm(self, rpn_features_to_crop, detection_boxes, matched_transcriptions, 
        detection_scores, num_detections, mode, sparse_code_target=None):
        # Reuse the second stage cropping as-is
        detection_model = self.detection_model
        flattened_detected_feature_maps = (
              detection_model._compute_second_stage_input_feature_maps(
                  rpn_features_to_crop, tf.expand_dims(detection_boxes, axis=0), stage='transcription'))


        with tf.variable_scope('Reshaping_cnn'):
            shape = flattened_detected_feature_maps.get_shape().as_list()  # [batch, height, width, features]
            print(shape)
            transposed = tf.transpose(flattened_detected_feature_maps, perm=[0, 2, 1, 3],
                                      name='transposed')  # [batch, width, height, features]
            conv_reshaped = tf.reshape(transposed, [-1, shape[2], shape[1] * shape[3]],
                                       name='reshaped')  # [batch, width, height x features]
        placeholder_features = {
            'corpus' : tf.zeros_like(detection_scores, dtype=tf.int32),
            'image_width': shape[2],
        }
        transcription_dict = self.lstm_layers(conv_reshaped, placeholder_features, mode, sparse_code_target)
        transcription_dict['labels'] = matched_transcriptions
        # transcription_dict['label_codes'] = sparse_code_target
        detections_dict = {}
        detections_dict['detection_boxes'] = detection_boxes
        detections_dict['detection_scores'] = detection_scores
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
        batch_size = predictions_dict['prob'].shape[1]
        with tf.control_dependencies([tf.less_equal(sparse_code_target.dense_shape[1], tf.reduce_max(tf.cast(seq_len_inputs, tf.int64)))]):
            loss_ctc = tf.nn.ctc_loss(labels=sparse_code_target,
                                      inputs=predictions_dict['prob'],
                                      sequence_length=seq_len_inputs,#tf.cast([seq_len_inputs] * batch_size, tf.int32),
                                      preprocess_collapse_repeated=False,
                                      ctc_merge_repeated=True,
                                      ignore_longer_outputs_than_inputs=True,  # returns zero gradient in case it happens -> ema loss = NaN
                                      time_major=True)
            loss_ctc = tf.reduce_mean(loss_ctc)
            # loss_ctc = tf.Print(loss_ctc, [loss_ctc], message='* Loss : ')
        return loss_ctc


    def compute_eval_ops(self, predictions_dict, detection_boxlist, sampled_indices, gt_transcriptions):
        with tf.name_scope('evaluation'):
            def encode_groundtruth(matched_transcriptions):
                matched_codes = self.str2code(matched_transcriptions)
                seq_lengths_labels = tf.bincount(tf.cast(matched_codes.indices[:, 0], tf.int32), #array of labels length
                                 minlength= tf.shape(matched_transcriptions)[0])
                target_chars = self.table_int2str.lookup(tf.cast(matched_codes, tf.int64))
                return get_words_from_chars(target_chars.values, seq_lengths_labels) 

            # Compute Precision
            matched_transcriptions = detection_boxlist.get_field(fields.BoxListFields.groundtruth_transcription)
            target_words = encode_groundtruth(matched_transcriptions)
            precision, precision_op = tf.metrics.accuracy(target_words, predictions_dict['words'][0], 
                name='precision')

            # Compute Character Error Rate
            # detection_boxlist.add_field(fields.BoxListFields.transcription, predictions_dict['words'][0])
            # sampled_boxlist = box_list_ops.boolean_mask(detection_boxlist, sampled_indices)
            # matched_transcriptions = sampled_boxlist.get_field(fields.BoxListFields.groundtruth_transcription)
            # sparse_code_target = self.str2code(matched_transcriptions)
            # matched_predictions = sampled_boxlist.get_field(fields.BoxListFields.transcription)
            # CER, CER_op = tf.metrics.mean(tf.edit_distance(sparse_code_pred[0], 
            #      tf.cast(sparse_code_target, dtype=tf.int64)), name='CER')

            # Compute Recall
            detection_boxlist.add_field(fields.BoxListFields.transcription, predictions_dict['words'][0])
            sampled_boxlist = box_list_ops.boolean_mask(detection_boxlist, sampled_indices)
            matched_transcriptions = sampled_boxlist.get_field(fields.BoxListFields.groundtruth_transcription)
            num_groundtruths = gt_transcriptions.get_shape().as_list()[0]
            num_matches = tf.shape(matched_transcriptions)[0]
            pad_size = [[0, num_groundtruths - num_matches]]
            matched_transcriptions = tf.pad(matched_transcriptions, 
                paddings=pad_size, constant_values='groundtruth')
            target_words = encode_groundtruth(matched_transcriptions)
            matched_predictions = sampled_boxlist.get_field(fields.BoxListFields.transcription)
            predicted_words = tf.pad(matched_predictions, 
                paddings=pad_size, constant_values='prediction')
            recall, recall_op = tf.metrics.accuracy(target_words, predicted_words, 
                name='recall')

            # Print to console
            precision = tf.Print(precision, [precision], message="Precision -- ")   
            recall = tf.Print(recall, [recall], message="Recall -- ")     
            eval_metric_ops = {
                'eval/precision' : (precision, precision_op),
                'eval/recall' : (recall, recall_op)
            }
            return eval_metric_ops

    # Code from crnn_fn
    def lstm_layers(self, feature_maps, features, mode, sparse_code_target=None):
        parameters = self.parameters

        logprob, raw_pred = deep_bidirectional_lstm(feature_maps, features['corpus'], params=parameters, summaries=False)
        # Compute seq_len from image width
        # n_pools = CONST.DIMENSION_REDUCTION_W_POOLING  # 2x2 pooling in dimension W on layer 1 and 2
        # seq_len_inputs = tf.divide(features['image_width'], n_pools, name='seq_len_input_op') - 1
        seq_len_inputs = tf.zeros_like(features['corpus']) + features['image_width']
        predictions_dict = {'prob': logprob,
                            'raw_predictions': raw_pred,
                            'seq_len_inputs': seq_len_inputs
                            }
           
        if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.TRAIN]:
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

                # tf.summary.text('predicted_words', predictions_dict['words'][0][:10])

        # Evaluation ops
        # --------------
        # if False and mode == tf.estimator.ModeKeys.EVAL:
        #     with tf.name_scope('evaluation'):
        #       seq_lengths_labels = tf.bincount(tf.cast(sparse_code_target.indices[:, 0], tf.int32), #array of labels length
        #                                          minlength= tf.shape(predictions_dict['prob'])[1])
        #       CER, CER_op = tf.metrics.mean(tf.edit_distance(sparse_code_pred[0], 
        #         tf.cast(sparse_code_target, dtype=tf.int64)), name='CER')
        #       # Convert label codes to decoding alphabet to compare predicted and groundtrouth words
        #       target_chars = table_int2str.lookup(tf.cast(sparse_code_target, tf.int64))
        #       target_words = get_words_from_chars(target_chars.values, seq_lengths_labels)
        #       accuracy, accuracy_op = tf.metrics.accuracy(target_words, predictions_dict['words'][0], 
        #         name='accuracy')
        #       CER = tf.Print(CER, [CER], message='-- CER : ')
        #       accuracy = tf.Print(accuracy, [accuracy], message='-- Accuracy : ')
        #       eval_metric_ops = {
        #                          'eval/accuracy': (accuracy, accuracy_op),
        #                          'eval/CER': (CER, CER_op),
        #                          }
        # else:                        
        #     eval_metric_ops = self.default_eval_op

        return predictions_dict

