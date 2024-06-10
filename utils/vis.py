import cv2

def draw_bboxes(image, bboxes, color=(0, 0, 255), type='xyxy', normalize=False):
    """Draw bounding boxes on the image
    
    Args:
        image (numpy.ndarray): the image to draw bounding boxes on
        bboxes (list): list of bounding boxes
        color (tuple): color of the bounding boxes
        type (str): type of the bounding boxes
        normalize (bool): whether the bounding boxes are normalized
    """
    for bbox in bboxes:
        if type == 'xyxy':
            c, x1, y1, x2, y2 = bbox
            if normalize:
                x1, y1, x2, y2 = int(x1 * image.shape[1]), int(y1 * image.shape[0]), int(x2 * image.shape[1]), int(y2 * image.shape[0])
        elif type == 'xywh':
            c, x1, y1, w, h = bbox
            if normalize:
                x1, y1, w, h = int(x1 * image.shape[1]), int(y1 * image.shape[0]), int(w * image.shape[1]), int(h * image.shape[0])
            x2, y2 = x1 + w, y1 + h
        else:
            raise ValueError('Invalid type')
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image