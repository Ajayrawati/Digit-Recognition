import cv2
import numpy as np
from keras.models import load_model
import sys
sys.stdout.reconfigure(encoding='utf-8')


model = load_model("Digit_Recognition.keras")
width, height = 280, 280
drawing_pad = np.zeros((height, width, 3), dtype=np.uint8)

drawing = False
last_point = (-1, -1)


def mouse_event(event, x, y, flags, param):
    global drawing, last_point, drawing_pad

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(drawing_pad, last_point, (x, y), (255, 255, 255), 15)
            last_point = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(drawing_pad, last_point, (x, y), (255, 255, 255), 15)
        last_point = (-1, -1)

cv2.namedWindow("Drawing Pad")
cv2.setMouseCallback("Drawing Pad", mouse_event)

while True:
    cv2.imshow("Drawing Pad", drawing_pad)
    
    key = cv2.waitKey(1)
    
    if key == ord('c'): 
        drawing_pad[:] = 0 
    
    elif key == ord('p'): 
        gray = cv2.cvtColor(drawing_pad, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 28, 28, 1))
        
        
        prediction = model.predict(reshaped)
        digit = np.argmax(prediction)
        print(f'Prediction: {digit}')
    
    elif key == ord('q'):  
        break

cv2.destroyAllWindows()
