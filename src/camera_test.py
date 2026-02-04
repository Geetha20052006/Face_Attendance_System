import cv2
import time

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return

    print("✅ Camera opened successfully")
    print("Press 'q' to exit")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Error: Failed to read frame")
            break

        # Get frame resolution
        height, width, channels = frame.shape
        print(f"Frame Resolution: {width} x {height}")

        # Convert frame to grayscale (test)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Display FPS on frame
        cv2.putText(frame, f"FPS: {int(fps)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Show frames
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Grayscale Frame", gray_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Exit key pressed")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera released and windows closed")

if __name__ == "__main__":
    main()
