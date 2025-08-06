#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tflite_runtime.interpreter as tflite
import csv


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier_base.tflite',
        num_threads=1,
        label_path='model/keypoint_classifier/keypoint_classifier_label.csv'
    ):
        self.interpreter = tflite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        with open(label_path, encoding='utf-8-sig') as f:
            self.labels = [row[0] for row in csv.reader(f)]

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index

    def predict(self, landmark_list):
        input_data = np.array([landmark_list], dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        return output_data  # softmax 확률 벡터
