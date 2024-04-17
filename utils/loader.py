import cv2

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img

if __name__ == "__main__":
    img = load_image("C://Users\lok20\OneDrive\_Master\MAIA-ERASMUS//2 Semester\Interdiscipilanry Project AIA_ML_DL\GRAZPEDWRI-DX\images_part1//0001_1297860395_01_WRI-L1_M014.png")
    print(img.dtype)
    print(img.shape)
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)