import cv2

def main():
    # Replace "camera_ip" with the LAN IP address of your WiFi camera
    camera_ip = "192.168.0.106"

    # Construct the camera URL
    # If your camera supports streaming over HTTP:
    camera_url = f"rtsp://{camera_ip}:554"
    # If your camera supports streaming over RTSP:
    # camera_url = f"rtsp://{camera_ip}"

    # Open the camera
    cap = cv2.VideoCapture(camera_url)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Unable to connect to the camera.")
        return

    # Loop to continuously read frames and display them
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame from the camera.")
            break

        # Display the frame
        cv2.imshow("Camera Feed", frame)

        # Check for keypress to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
