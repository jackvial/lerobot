import os
import cv2
import numpy as np
import time

print(cv2.__file__)

# Debugging Environment Variables
print("Setting environment variables...")
os.environ["QT_X11_NO_MITSHM"] = "1"  # Avoid shared memory issues
os.environ["DISPLAY"] = os.getenv("DISPLAY", ":10.0")  # Ensure DISPLAY is set
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"  # Force software rendering

print("Environment variables:")
print(f"DISPLAY = {os.environ['DISPLAY']}")
print(f"QT_X11_NO_MITSHM = {os.environ['QT_X11_NO_MITSHM']}")
print(f"LIBGL_ALWAYS_SOFTWARE = {os.environ['LIBGL_ALWAYS_SOFTWARE']}")

# Create a simple test image
print("Creating test image...")
image = np.zeros((300, 300, 3), dtype=np.uint8)
image[:] = (0, 255, 0)  # Fill with green color

# Try displaying the image
print("Attempting to display the image using cv2.imshow...")
cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
try:
    cv2.imshow("Test Window", image)
    print("cv2.imshow succeeded.")
except Exception as e:
    print(f"Error during cv2.imshow: {e}")

# Add delay to keep the window open
print("Waiting for 1 second using cv2.waitKey...")
key = cv2.waitKey(1000)
print(f"cv2.waitKey returned: {key}")

print("Closing all windows...")
cv2.destroyAllWindows()

import threading
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

# Use 'Agg' backend for off-screen rendering
matplotlib.use('Agg')

def render_in_background():
    print("Background thread started for rendering...")
    for i in range(5):  # Render 5 frames
        # Generate some test data
        data = np.random.rand(300, 300, 3)
        
        # Create a plot (off-screen)
        fig, ax = plt.subplots()
        ax.imshow(data)
        ax.set_title(f"Frame {i+1}")
        
        # Save the frame to a file
        output_file = f"frame_{i+1}.png"
        fig.savefig(output_file)
        plt.close(fig)
        
        print(f"Rendered and saved: {output_file}")
        time.sleep(1)  # Simulate some work

# Start rendering in a background thread
render_thread = threading.Thread(target=render_in_background)
render_thread.start()

print("Main thread is free to continue execution...")
render_thread.join()
print("Background rendering completed.")