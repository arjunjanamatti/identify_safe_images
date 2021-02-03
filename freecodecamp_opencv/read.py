import cv2 as cv

# read the image
img = cv.imread(filename = 'C:/Users/Arjun Janamatti/Documents/image_classification/nude_sexy_safe_v1_x320/testing/safe/0AFBAF6E-AC8F-4F86-B131-26AF632BF574.jpg')
print('Shape of the image: ', img.shape)

# display the read image
cv.imshow(winname='image_window', mat=img)
cv.waitKey(0)

# reading videos
capture = cv.VideoCapture('C:/Users/Arjun Janamatti/PycharmProjects/jeeva_project/upload_videos/video_2.mp4')

while True:
    isTrue, frame = capture.read()

    cv.imshow('Video play', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()
