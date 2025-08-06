import cv2
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite

CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;?@[\\]^_`{|}~ "

interpreter = tflite.Interpreter(model_path="recognizer_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    decoded, _ = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
    decoded = decoded[0][0].numpy()
    return ''.join(CHARSET[idx] for idx in decoded.flatten() if 0 <= idx < len(CHARSET))

def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 31))
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, 31, 200, 1)

def recognize_single_text(roi):
    input_data = preprocess_roi(roi)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    return decode_prediction(y_pred)

def enhance_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(eq, -1, sharpen_kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

def decode_predictions(scores, geometry, min_confidence):
    (numRows, numCols) = scores.shape[2:4]
    rects, confidences = [], []
    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        for x in range(numCols):
            if scoresData[x] < min_confidence:
                continue
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = anglesData[x]
            cos, sin = np.cos(angle), np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX, startY = int(endX - w), int(endY - h)
            rects.append([startX, startY, endX, endY])
            confidences.append(float(scoresData[x]))
    return rects, confidences

def detect_text_boxes_east(roi, east_model_path="frozen_east_text_detection.pb", min_confidence=0.5):
    (H, W) = roi.shape[:2]
    newW, newH = 640, 640
    rW, rH = W / float(newW), H / float(newH)
    blob = cv2.dnn.blobFromImage(roi, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net = cv2.dnn.readNet(east_model_path)
    net.setInput(blob)
    scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    rects, confidences = decode_predictions(scores, geometry, min_confidence)
    boxes = cv2.dnn.NMSBoxes(rects, confidences, min_confidence, 0.4)
    results = []
    if len(boxes) > 0:
        for i in boxes.flatten():
            startX, startY, endX, endY = rects[i]
            results.append((
                max(0, int(startX * rW)),
                max(0, int(startY * rH)),
                max(0, int(endX * rW)),
                max(0, int(endY * rH))
            ))
    return results

# ✅ 최종 통합 OCR 호출 함수
def run_ocr_on_roi(image_path, x1, y1, x2, y2, show_result=False):
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 로딩 실패: {image_path}")
        return []

    roi = image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)].copy()
    roi = enhance_roi(roi)

    try:
        boxes = detect_text_boxes_east(roi, min_confidence=0.3)
        texts = []
        result_image = roi.copy()

        for i, (sx, sy, ex, ey) in enumerate(boxes):
            if ex - sx < 5 or ey - sy < 5:
                continue
            cropped = roi[sy:ey, sx:ex]
            try:
                text = recognize_single_text(cropped)
                texts.append(text)
                if show_result:
                    cv2.rectangle(result_image, (sx, sy), (ex, ey), (0, 255, 0), 2)
                    cv2.putText(result_image, text, (sx, sy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            except Exception as e:
                texts.append('[인식 실패]')
        
        if show_result:
            cv2.imshow("OCR Result", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return texts
    except Exception as e:
        print(f"OCR 실패: {e}")
        return []
