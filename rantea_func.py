import cv2
import numpy as np
import onnxruntime
from keras.applications.resnet import preprocess_input
from classes import WBsRGB as wb_srgb


def object_detection(image_path, model_path):
    ort_session = onnxruntime.InferenceSession(model_path)
    model_inputs = ort_session.get_modelmeta()
    model_inputs = ort_session.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]

    model_output = ort_session.get_outputs()
    output_names = [model_output[i].name for i in range(len(model_output))]
    input_shape = model_inputs[0].shape
    # Read Image
    image = cv2.imread(image_path)
    # Image shape
    image_height, image_width = image.shape[:2]
    # Input shape
    input_height, input_width = input_shape[2:]
    # Convert image bgr to rgb
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize image to input shape
    resized = cv2.resize(image_rgb, (input_width, input_height))

    # Scale input pixel value to 0 to 1
    input_image = resized / 255.0
    input_image = input_image.transpose(2, 0, 1)
    input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
    input_tensor.shape

    outputs = ort_session.run(output_names, {input_names[0]: input_tensor})[0]

    predictions = np.squeeze(outputs).T

    conf_thresold = 0.5
    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_thresold, :]
    scores = scores[scores > conf_thresold]

    # Get bounding boxes for each object
    boxes = predictions[:, :4]

    # rescale box
    input_shape = np.array([input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([image_width, image_height, image_width, image_height])
    boxes = boxes.astype(np.int32)

    indices = nms(boxes, scores, 0.3)

    # Get the bounding box with the highest score
    max_score_index = np.argmax(scores[indices])
    max_score_box = (
        xywh2xyxy(boxes[indices[max_score_index]]).round().astype(np.int32).tolist()
    )

    # Crop the image using the bounding box with the highest score
    x1, y1, x2, y2 = max_score_box
    cropped_image = image_rgb[y1:y2, x1:x2]

    cropped_image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
    result_img_bgr = white_balance_image(cropped_image_bgr)
    result_img = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)

    result_img = (result_img * 255).astype(np.uint8)

    return cropped_image


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def predict(img, loaded_best_model):

    prod_output = {"class": None, "prob": 0}

    if img is None:
        return prod_output

    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)

    # Label array
    labels = {
        0: "BOPF3",
        1: "BOPF4",
        2: "BP3",
        3: "BP4",
        4: "BT2",
        5: "BT3",
        6: "BT4",
        7: "DUST3",
        8: "DUST4",
        9: "PF3",
        10: "PF4",
    }

    # Get the Predicted Label for the loaded Image
    p = loaded_best_model.predict(np.expand_dims(img, axis=0), verbose=0)

    predicted_class = labels[np.argmax(p[0], axis=-1)]

    prod_output["prob"] = np.max(p[0], axis=-1)
    prod_output["class"] = predicted_class
    return prod_output

def white_balance_image(image_array, upgraded_model=0, gamut_mapping=2):
# create an instance of the WB model
    wbModel = wb_srgb.WBsRGB(gamut_mapping=gamut_mapping, upgraded=upgraded_model)
    outImg = wbModel.correctImage(image_array)  # white balance it
    return outImg