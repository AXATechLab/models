import tensorflow as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch
from nets import resnet_utils
from nets import resnet_v1
from functools import partial

import sys
sys.path.append("/notebooks/Transcription/tf-crnn/")
from tf_crnn.model import deep_cnn

slim = tf.contrib.slim

class FasterRCNNCNet(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Faster R-CNN Resnet V1 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0):
    """Constructor.

    Args:
      architecture: Architecture name of the Resnet V1 model.
      resnet_model: Definition of the Resnet V1 model.
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    self.has_coupled_domains = False
    super(FasterRCNNCNet, self).__init__(
        is_training, first_stage_features_stride, batch_norm_trainable,
        reuse_weights, weight_decay)

  def preprocess(self, resized_inputs):
    """Faster R-CNN Resnet V1 preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.

    """
    # channel_means = [123.68, 116.779, 103.939]
    # return resized_inputs - [[channel_means]]
    return resized_inputs

  def _extract_proposal_features(self, preprocessed_inputs, scope):
    """Extracts first stage RPN features.

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float32 tensor
        representing a batch of images.
      scope: A scope name.

    Returns:
      rpn_feature_map: A tensor with shape [batch, height, width, depth]
      activations: A dictionary mapping feature extractor tensor names to
        tensors

    Raises:
      InvalidArgumentError: If the spatial size of `preprocessed_inputs`
        (height or width) is less than 33.
      ValueError: If the created network is missing the required activation.
    """
    return deep_cnn(preprocessed_inputs, self._is_training, summaries=False, on_document=True), None


  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features.

    Args:
      proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      scope: A scope name (unused).

    Returns:
      proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
    """
    with tf.variable_scope('resnet_v1_101', reuse=self._reuse_weights):
      with slim.arg_scope(
          resnet_utils.resnet_arg_scope(
              batch_norm_epsilon=1e-5,
              batch_norm_scale=True,
              weight_decay=self._weight_decay)):
        with slim.arg_scope([slim.batch_norm],
                            is_training=self._train_batch_norm):
          blocks = [
              resnet_utils.Block('block4', resnet_v1.bottleneck, [{
                  'depth': 2048,
                  'depth_bottleneck': 512,
                  'stride': 1
              }] * 3)
          ]
          proposal_classifier_features = resnet_utils.stack_blocks_dense(
              proposal_feature_maps, blocks)
    return proposal_classifier_features
