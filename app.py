import cv2
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import Response

app = FastAPI()

width, height = 107, 48
PIXEL_THRESHOLD = 800
posList = [(50, 150), (200, 150), (350, 150), (50, 350), (200, 350)]


@app.get("/")
def read_root():
    return {"message": "hello world"}


@app.get("/parking")
def get_parking(cars: int = Query(2, ge=0, le=5)):
    # 1. Create the dark pavement foundation matrix
    img = np.ones((600, 800, 3), dtype=np.uint8) * 50
    for pos in posList:
        x, y = pos
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 255, 255), 2)

    # Fill slot coordinates using custom blocks based on requested parameter metric count
    for i in range(min(cars, len(posList))):
        x, y = posList[i]
        cv2.rectangle(img, (x + 2, y + 2), (x + width - 2, y + height - 2), (180, 105, 255), -1)

    # 2. Process image boundaries using morphological filters
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    imgDilate = cv2.dilate(imgMedian, np.ones((3, 3), np.uint8), iterations=1)

    # 3. Analyze edge counts and build semi-transparent overlay overlays
    overlay = img.copy()
    free_counter = 0

    for pos in posList:
        x, y = pos
        imgCrop = imgDilate[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count > PIXEL_THRESHOLD:
            color, thickness = (0, 255, 0), 3
            free_counter += 1
        else:
            color, thickness = (0, 0, 255), 2

        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
        cv2.rectangle(img, (x, y), (x + width, y + height), color, thickness)
        cv2.putText(img, str(count), (x, y + height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    # Scoreboard metric system overlay positioning metrics
    cv2.rectangle(img, (20, 20), (250, 80), (0, 0, 0), -1)
    cv2.putText(img, f'FREE: {free_counter}/5', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    img_resized = cv2.resize(img, (640, 480))

    # 4. Return structural streaming frame data packets
    _, buffer = cv2.imencode('.jpg', img_resized)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")
