import cv2
import numpy as np
import math
from imutils.video.count_frames import count_frames_manual
import progressbar as progressbar
import glob
from colour import compute_color_probabilities, color_select, increase_brightness_and_saturation, ColorPalette
from utils import randomized_grid

# The input video to be processed
video = cv2.VideoCapture("C:\\Users\\test1\\Pictures\\Camera Roll\\knife.mp4")
# Counting the numbr of frames in the video
total = count_frames_manual(video)

# Input video to be processed
cap = cv2.VideoCapture("C:\\Users\\test1\\Pictures\\Camera Roll\\knife.mp4")
count = 0


# Read the first frame
ret, frame1 = cap.read()

# If input video is too small, exit
if math.ceil(min(frame1.shape[0], frame1.shape[1])) < 350:
    print("input image too small")
    exit()

# If the image is too large, downsample it
if math.ceil(max(frame1.shape)) > 2500:
    imageshape = 2500 / math.ceil(max(frame1.shape))
    newX, newY = frame1.shape[1] * imageshape, frame1.shape[0] * imageshape
    frame1 = cv2.resize(frame1, (int(newX), int(newY)))

prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while count < total-1:
    # Read the next frame and convert to black and white
    ret, image = cap.read()

    # If the image is too large, downsample it
    if math.ceil(max(image.shape)) > 2500:
        imageshape = 2500 / math.ceil(max(image.shape))
        newX, newY = image.shape[1] * imageshape, image.shape[0] * imageshape
        image = cv2.resize(image, (int(newX), int(newY)))
    next = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow between previous and next
    flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    ang * 180 / np.pi / 2

    prev = next
    height, width, depth = image.shape

    # Apply the bilateral filter
    dst = cv2.bilateralFilter(image, 9, 75, 75)

    # Increase the brightness and saturation of the image
    bgr = increase_brightness_and_saturation(dst, 10)
    stroke_scale = int(math.ceil(min(bgr.shape) / 10000))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Generate the colour palette
    palette = ColorPalette.from_image(bgr, 11)
    blank_image = np.zeros((height, width, 3), np.uint8)
    blank_image[:, 0:width] = (255, 255, 255)

    # If the first frame is being processed then generate a new grid, otherwise use the old grid
    if count == 0:
        grid2 = randomized_grid(blank_image.shape[0], blank_image.shape[1], 3)
    bar = progressbar.ProgressBar()

    for h in bar(range(0, len(grid2), 100000)):
        # get the pixel colours at each point of the grid
        pixels = np.array([bgr[x[0], x[1]] for x in grid2[h:min(h + 100000, len(grid2))]])

        # Compute the probabilities for each color in the palette, lower values of k means more randomness
        color_probabilities = compute_color_probabilities(pixels, palette, k=6)

        # Paint the first layer of strokes
        for i, (y, x) in enumerate(grid2[h:min(h + 100000, len(grid2))]):

            # choose the colour for the stroke
            color = color_select(color_probabilities[i], palette)

            # choose the length of the stroke
            length = int(round(stroke_scale * 2))

            # draw the brush stroke
            a = x
            b = y

            # Convert angles from radians to degrees
            angle = ang[round(y), round(x)] * 57.2958

            # Choose the direction of the stroke
            # If there is motion, the stroke will follow the motion
            # If not, then the angle of the stroke is 135 degrees
            if mag[round(y), round(x)] < 1.8:
                angle = 135

            # Finally, draw the stroke
            cv2.ellipse(blank_image, (x, y), (length, stroke_scale), angle, 0, 360, color, -1, cv2.LINE_AA)

        # Paint the second layer of strokes
        for i, (y, x) in enumerate(grid2[h:min(h + 100000, len(grid2))]):
            # choose the colour for the stroke
            color = color_select(color_probabilities[i], palette)

            # choose the length of the stroke
            length = int(round(stroke_scale * 2))

            # draw the brush stroke
            a = x
            b = y

            # Convert angles from radians to degrees
            angle = ang[round(y), round(x)] * 57.2958

            # Choose the direction of the stroke
            # If there is motion, the stroke will follow the motion
            # If not, then the angle of the stroke is 135 degrees
            if mag[round(y), round(x)] < 1.8:
                angle = 135

            # Finally, draw the stroke
            cv2.ellipse(blank_image, (x, y), (length, stroke_scale), angle, 0, 360, color, -1, cv2.LINE_AA)

    count += 1
    print(count, "/", total, " frames rendered")

    # Saving the frame to file
    cv2.imwrite("C:\\Users\\test1\\Pictures\\experiment\\a%03d.jpg" % count, blank_image)

    # Showing the rendered frame
    cv2.imshow('Frame', blank_image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Adding the frames together to make the animation
print("adding frames together")
img_array = []
for filename in glob.glob("C:\\Users\\test1\\Pictures\\experiment\\*.jpg"):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('C:\\Users\\test1\\Pictures\\Camera Roll\\demope.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 12, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
