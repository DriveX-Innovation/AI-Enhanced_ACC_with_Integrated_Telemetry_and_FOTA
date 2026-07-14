import cv2
import numpy as np
import ncnn

def main():
    print("Loading NCNN Engine and Optimized Multi-Task Model...")
    
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False 
    net.opt.num_threads = 4  
    
    # Matching your exact file names
    net.load_param("models/yolopv2.param")
    net.load_model("models/yolopv2.bin")

    print("Model loaded successfully! Starting camera...")
    
    # Camera hardware optimizations
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    mean_vals = [123.675, 116.28, 103.53]
    norm_vals = [0.017125, 0.017507, 0.017429]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        orig_h, orig_w = frame.shape[:2]

        # --- PREPROCESSING ---
        mat_in = ncnn.Mat.from_pixels_resize(
            frame,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            orig_w, orig_h,
            640, 640
        )
        mat_in.substract_mean_normalize(mean_vals, norm_vals)

        # --- INFERENCE ---
        ex = net.create_extractor()
        ex.input("input", mat_in) 

        # --- THE MASK HUNTER ---
        # Automatically find and reconstruct the masks, ignoring NCNN's internal renaming
        masks = []
        for blob_name in ["677", "769", "det0", "det1", "det2"]:
            ret_val, mat = ex.extract(blob_name)
            if ret_val == 0:
                arr = np.array(mat)
                
                # Check for 1-Channel flattened mask
                if arr.size == 409600:
                    masks.append(arr.reshape(640, 640))
                # Check for 2-Channel flattened mask (Background vs Foreground)
                elif arr.size == 819200:
                    arr_2d = arr.reshape(2, 640, 640)
                    masks.append(arr_2d[1]) # Grab the foreground channel

        # Ensure the engine successfully yielded both the Lane and Area masks
        if len(masks) < 2:
            print(f"Searching for AI masks... Found {len(masks)}. Skipping frame.")
            continue

        lane_mask_np = masks[0]
        area_mask_np = masks[1]

        # --- POST-PROCESSING ---
        # Now that they are guaranteed 2D matrices, OpenCV will resize them flawlessly
        lane_mask_resized = cv2.resize(lane_mask_np, (orig_w, orig_h))
        area_mask_resized = cv2.resize(area_mask_np, (orig_w, orig_h))

        # Thresholding
        lane_binary = lane_mask_resized > 0.5
        area_binary = area_mask_resized > 0.5

        color_overlay = np.zeros_like(frame)

        # Apply Colors 
        # (If your lane draws green and your area draws red, just swap these two arrays!)
        color_overlay[area_binary] = [0, 200, 0]    # Green for Drivable Area
        color_overlay[lane_binary] = [0, 0, 255]    # Red for Lane Lines

        # Blend the colored mask seamlessly onto the live camera feed
        frame = cv2.addWeighted(color_overlay, 0.5, frame, 1.0, 0)

        # --- DISPLAY ---
        cv2.imshow("Multi-Task ADAS Dashboard", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
