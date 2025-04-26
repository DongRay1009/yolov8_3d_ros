def draw_bounding_boxes(image, boxes, scores, classes, class_names):
    for box, score, cls in zip(boxes, scores, classes):
        # You may need to adjust confidence threshold based on your application
        if score < 0.5:  # Threshold for displaying boxes
            continue
        x1, y1, x2, y2 = box
        # You can customize different colors for different object classes
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{class_names[cls]}: {score:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def visualize_detections(image, detections, class_names):
    boxes = detections['boxes']
    scores = detections['scores']
    classes = detections['classes']
    return draw_bounding_boxes(image, boxes, scores, classes, class_names)