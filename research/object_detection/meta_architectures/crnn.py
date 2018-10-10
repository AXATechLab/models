import tensorflow as tf

from core import box_list, box_list_ops
from core import standard_fields as fields
import sys
sys.path.append("~/Documents/Git/tf-crnn/")
from src.model import deep_bidirectional_lstm

class CRNN(object):

	def __init__(self, params, detection_model, target_assigner):
		self.parameters = params
		self.detection_model = detection_model
		self.target_assigner = target_assigner

	# This is in the same fashion as predict_third_stage's inference
	# TODO: use parameters instead of copying FasterRCNN's
	def predict(self, prediction_dict, labels, true_image_shapes):
		# Postprocess FasterRCNN stage 2
		detections_dict = self.detection_model._postprocess_box_classifier(
			prediction_dict['refined_box_encodings'],
			prediction_dict['class_predictions_with_background'],
			prediction_dict['proposal_boxes'],
			prediction_dict['num_proposals'],
			true_image_shapes)
		prediction_dict.update(detections_dict)
		detection_boxes = detections_dict[
		  fields.DetectionResultFields.detection_boxes][0]
		detection_scores = detections_dict[
			fields.DetectionResultFields.detection_scores][0]
		detection_transcriptions = None

		num_detections = detections_dict[
			fields.DetectionResultFields.num_detections][0]
		rpn_features_to_crop = prediction_dict['rpn_features_to_crop']

	    if detection_model._is_training:
			gt_boxlists, gt_classes, _, gt_weights, gt_transcriptions = detection_model._format_groundtruth_data(true_image_shapes, 
				stage='transcription')

			detection_boxlist = box_list_ops.to_absolute_coordinates(box_list.BoxList(detection_boxes))
	      	detection_boxlist.add_field(fields.BoxListFields.scores, detection_scores)

			(_, cls_weights, _, _, match) = self.target_assigner.assign(detection_boxlist, 
				gt_boxlists[0], gt_classes[0],         
				unmatched_class_label=tf.constant(
	            [1] + detection_model._num_classes * [0], dtype=tf.float32),
	         	groundtruth_weights=gt_weights[0])

			matching = match.match_results() # indices of matched groundtruths, or negative number if unmatched
			detection_transcriptions = tf.constant([None] * num_detections, dtype=tf.string)
			for dt_ind, gt_ind in enumerate(matching):
				if gt_ind < 0:
					continue
				detection_transcritpions[dt_ind] = gt_transcriptions[0][gt_ind]
	      	detection_boxlist.add_field(fields.BoxListFields.transcription, detection_transcriptions)


			positive_indicator = tf.greater(matching, -1)
			valid_indicator = tf.logical_and(
		        tf.range(detection_boxlist.num_boxes()) < num_detections,
		        cls_weights > 0
		    )
			sampled_indices = detection_model._second_stage_sampler.subsample(
		        valid_indicator,
		        detection_model._second_stage_batch_size,
		        positive_indicator)

	 		sampled_boxlist = box_list_ops.boolean_mask(detection_boxlist, sampled_indices)

			sampled_padded_boxlist = box_list_ops.pad_or_clip_box_list(
	          sampled_boxlist,
	          num_boxes=detection_model._second_stage_batch_size)

			# Update detections with matched detections
			detection_boxes = sampled_padded_boxlist.get()
			detection_transcriptions = sampled_padded_boxlist.get_field(fields.BoxListFields.transcription)
			detection_scores = sampled_padded_boxlist.get_field(fields.BoxListFields.scores)
	      	num_detections = tf.minimum(sampled_boxlist.num_boxes(),
	          detection_model._second_stage_batch_size)

 		# Reuse the second stage cropping as-is
      	flattened_detected_feature_maps = (
	          detection_model._compute_second_stage_input_feature_maps(
	              rpn_features_to_crop, tf.expand_dims(detection_boxes, axis=0)))

      	self.lstm_layers()

	# Code from crnn_fn
	def lstm_layers(self, features, labels):
		parameters = self.parameters
		logprob, raw_pred = deep_bidirectional_lstm(conv, features['corpus'], params=parameters, summaries=False)

	    # Compute seq_len from image width
	    n_pools = CONST.DIMENSION_REDUCTION_W_POOLING  # 2x2 pooling in dimension W on layer 1 and 2
	    seq_len_inputs = tf.divide(features['image_width'], n_pools, name='seq_len_input_op') - 1

	    predictions_dict = {'prob': logprob,
	                        'raw_predictions': raw_pred
	                        }

	    if not mode == tf.estimator.ModeKeys.PREDICT:
	        # Alphabet and codes
	        keys = [c for c in parameters.alphabet.encode('latin1')]
	        values = parameters.alphabet_codes

	        # Convert string label to code label
	        with tf.name_scope('str2code_conversion'):
	            table_str2int = tf.contrib.lookup.HashTable(
	                tf.contrib.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.int64, value_dtype=tf.int64), -1)
	            splitted = tf.string_split(labels, delimiter='')
	            values_int = tf.cast(tf.squeeze(tf.decode_raw(splitted.values, tf.uint8)), tf.int64)
	            codes = table_str2int.lookup(values_int)
	            codes = tf.cast(codes, tf.int32)
	            sparse_code_target = tf.SparseTensor(splitted.indices, codes, splitted.dense_shape)

	        seq_lengths_labels = tf.bincount(tf.cast(sparse_code_target.indices[:, 0], tf.int32), #array of labels length
	                                         minlength= tf.shape(predictions_dict['prob'])[1])

	        # Loss
	        # ----
	        # >>> Cannot have longer labels than predictions -> error

	        with tf.control_dependencies([tf.less_equal(sparse_code_target.dense_shape[1], tf.reduce_max(tf.cast(seq_len_inputs, tf.int64)))]):
	            loss_ctc = tf.nn.ctc_loss(labels=sparse_code_target,
	                                      inputs=predictions_dict['prob'],
	                                      sequence_length=tf.cast(seq_len_inputs, tf.int32),
	                                      preprocess_collapse_repeated=False,
	                                      ctc_merge_repeated=True,
	                                      ignore_longer_outputs_than_inputs=True,  # returns zero gradient in case it happens -> ema loss = NaN
	                                      time_major=True)
	            loss_ctc = tf.reduce_mean(loss_ctc)
	            loss_ctc = tf.Print(loss_ctc, [loss_ctc], message='* Loss : ')
        if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.TRAIN]:
	        with tf.name_scope('code2str_conversion'):
	            keys = tf.cast(parameters.alphabet_decoding_codes, tf.int64)
	            values = [c for c in parameters.alphabet_decoding]
	            table_int2str = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), '?')

	            sparse_code_pred, log_probability = tf.nn.ctc_beam_search_decoder(predictions_dict['prob'],
	                                                                              sequence_length=tf.cast(seq_len_inputs, tf.int32),
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

	            tf.summary.text('predicted_words', predictions_dict['words'][0][:10])


