
import os
import sys

import numpy as np
import argparse
import cv2
import tensorflow as tf
from object_detection import eval_util as visualizer

from object_detection.metrics import coco_tools

from glob import glob

def visualize_predictions(annot_paths, graph_path, output_dir, max_num_evaluations, max_num_visualizations,
    max_num_predictions):
    """ Return a set of `results` for all images in the given list
    This is a dictionary conforming to the requirements of
    `evaluate_detection_results_pascal_voc`"""
    if len(annot_paths) == 0:
        raise Exception("No images found.")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        detection_sess = tf.Session(graph=detection_graph)

        ''' Make output directories '''
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(output_dir + "images"):
            os.makedirs(output_dir + "images")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir + "log")

        ''' Store data here for metric evaluation '''
        list_dict = {
            'image_ids' : [],
            'dt_boxes_list' : [],
            'dt_classes_list' : [],
            'dt_scores_list' : [],
            'category_index' : [{'id' : 1, 'name' : 'HW' }]
        }
        model_dict = {
            'image_tensor' : detection_graph.get_tensor_by_name('image_tensor:0'),
            'template_id': detection_graph.get_tensor_by_name('template_id:0'),
            'detection_boxes' : detection_graph.get_tensor_by_name('detection_boxes:0'),
            'detection_scores' : detection_graph.get_tensor_by_name('detection_scores:0'),
            'detection_classes' : detection_graph.get_tensor_by_name('detection_classes:0'),
            'detection_corpora': detection_graph.get_tensor_by_name('detection_corpora:0'),
            'num_detections' : detection_graph.get_tensor_by_name('num_detections:0'),
            'words' : detection_graph.get_tensor_by_name('words:0'),
            'score' : detection_graph.get_tensor_by_name('score:0'),
        }
        params = {
            'max_num_visualizations' : max_num_visualizations,
            'max_num_predictions' : max_num_predictions,
            'output_dir' : output_dir
        }

        # Initialize tables
        detection_sess.run([detection_graph.get_operation_by_name('key_value_init'),
        	detection_graph.get_operation_by_name('key_value_init_1')])


        count = 0
        for image_annot_path in annot_paths:
            if max_num_evaluations >= 0 and count >= max_num_evaluations:
                break
            print("Input File:", image_annot_path)
            key = os.path.splitext(os.path.split(image_annot_path)[1])[0]
            image_np = cv2.imread(image_annot_path, cv2.IMREAD_COLOR)
            b,g,r = cv2.split(image_np)
            image_np = cv2.merge([r,g,b])

            visualize_single_example(image_np, key, detection_sess, params, list_dict, model_dict, count)
            count = count + 1

    detection_sess.close()


def visualize_single_example(image_np, key, detection_sess, params, list_dict, model_dict, count):
    image_np_expanded = np.expand_dims(image_np, axis=0)    # [1, H, W, C]
    print(count)
    # A template id of -1 means not available. This way the assertion on multiple templates will fail (they are
    # not supported for now). A model with a single template will run.
    tid = np.array([-1], dtype=np.int64)
    try:
        dt_boxes, dt_scores, dt_classes, dt_corpora, transcriptions, transcription_scores = detection_sess.run([
            model_dict['detection_boxes'],
            model_dict['detection_scores'],
            model_dict['detection_classes'],
            model_dict['detection_corpora'],
            model_dict['words'],
            model_dict['score']
            ],
            feed_dict={model_dict['image_tensor']: image_np_expanded, model_dict['template_id']: tid})
    except tf.errors.InvalidArgumentError as e:
        print("InvalidArgumentError occurred, are you using a model with multiple templates? (unsupported).")
        raise e

    ''' Get rid of the first dimension, de-normalize and format for visualization
        (as expected in eval_util.visualize_detection_results) '''
    dt_boxes = dt_boxes.reshape((-1, 4))
    dt_scores = dt_scores.reshape(-1)
    dt_classes = dt_classes.reshape(-1)
    dt_corpora = dt_corpora.reshape(-1)
    transcriptions = transcriptions.reshape(-1)
    transcription_scores = transcription_scores.reshape(-1)
    dt_boxes[:, [0, 2]] *= image_np.shape[0]
    dt_boxes[:, [1, 3]] *= image_np.shape[1]
    result_dict_for_single_example = {
        'original_image' : image_np_expanded,
        'key' : key,
        'detection_boxes' : dt_boxes,
        'detection_scores' : dt_scores,
        'detection_classes' : dt_classes,
    }

    ''' Draw and write out detections '''
    if params['max_num_visualizations'] < 0 or count < params['max_num_visualizations']:
        visualizer.visualize_detection_results(result_dict_for_single_example, key, -1, list_dict['category_index'],
            summary_dir=(params['output_dir'] + "log/"), export_dir=(params['output_dir'] + "images"), show_groundtruth=False,
            max_num_predictions=params['max_num_predictions'], transcriptions=transcriptions, detection_corpora=dt_corpora,
            keep_image_id_for_visualization_export=False,
            min_score_thresh=.5)

    ''' Store example for metric evaluation '''
    list_dict['image_ids'].append(key)
    list_dict['dt_scores_list'].append(dt_scores)
    list_dict['dt_classes_list'].append(dt_classes)
    list_dict['dt_boxes_list'].append(dt_boxes)

def evaluate_COCO_metrics(list_dict):
    ''' Compute COCO metrics. '''
    groundtruth_dict = coco_tools.ExportGroundtruthToCOCO(
      list_dict['image_ids'], list_dict['gt_boxes_list'], list_dict['gt_classes_list'],
      list_dict['category_index'], output_path=None)

    detections_list = coco_tools.ExportDetectionsToCOCO(
      list_dict['image_ids'], list_dict['dt_boxes_list'], list_dict['dt_scores_list'],
      list_dict['dt_classes_list'], list_dict['category_index'], output_path=None)

    groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
    detections = groundtruth.LoadAnnotations(detections_list)
    evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections,
                                         agnostic_mode=False)
    return evaluator.ComputeMetrics()


def main():
    parser = argparse.ArgumentParser(description='Visualize detections.')
    parser.add_argument('--input_dir', dest='input',
                    help='Directory of xml annotations or tfrecords', required=True)
    parser.add_argument('--graph_path', dest='graph',
                    help='Path to frozen graph. We assume the model returns detections in normalized coordinates and the input image is RGB.', required=True)
    parser.add_argument('--out_dir', dest='out',
                    help='Directory of output visualizations', required=True)
    # parser.add_argument('--tfrecords', help="Whether to visualize tfrecords instead of xml files",
    #                 action='store_const', dest='tfrecords', const=True, default=False)
    parser.add_argument('--max_num_evaluations', help="The maximum number of evaluated images allowed. Negative number means infinity",
                    type=int, dest='max_num_evaluations', default=-1)
    parser.add_argument('--max_num_visualizations', help="The maximum number of visualizations allowed. Note that this parameter affects only visualizations, not the number of computed predictions. Negative number means infinity",
                    type=int, dest='max_num_visualizations', default=-1)
    parser.add_argument('--max_num_predictions', help="The maximum number of predictions to be visualized per image.",
                    type=int, dest='max_num_predictions', default=100)

    args = parser.parse_args()

    annot_paths = glob(args.input + '*.png')
    graph_path = args.graph
    output_dir = args.out

    '''Run predictions on the dataset'''
    visualize_predictions(annot_paths, graph_path, output_dir,
        max_num_evaluations=args.max_num_evaluations,
        max_num_visualizations=args.max_num_visualizations,
        max_num_predictions=args.max_num_predictions)


if __name__ == '__main__':
    main()