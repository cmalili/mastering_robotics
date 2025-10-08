# webcam_workflow_sdk.py
import cv2, os, tempfile, base64, io, numpy as np
from PIL import Image
from inference_sdk import InferenceHTTPClient

API_URL   = "http://localhost:9001"
WORKSPACE = "chris-hub"
WORKFLOW  = "detect-and-classify-3"   # change to your workflow id
API_KEY   = "1HhSNS3VWex8YfHgeGzJ"  # put your key if your container wasn't started with ROBOFLOW_API_KEY

def get_output(result, name):
    if isinstance(result, dict):
        return result.get(name)
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and name in item:
                return item[name]
    return None

def draw_boxes_bgr(img_bgr, preds):
    for p in preds:
        cx, cy, w, h = p["x"], p["y"], p["width"], p["height"]
        x1, y1 = int(cx - w/2), int(cy - h/2)
        x2, y2 = int(cx + w/2), int(cy + h/2)
        cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img_bgr, f'{p["class"]}:{p["confidence"]:.2f}',
                    (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return img_bgr

def main():
    client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

    # one reusable temp path to avoid creating files each frame
    tmp_path = os.path.join(tempfile.gettempdir(), "rf_frame.jpg")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (try index 1 or 2).")

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # write current frame to a stable temp file (SDK accepts file paths)
        cv2.imwrite(tmp_path, frame)

        try:
            result = client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW,
                images={"image": tmp_path}  # <- pass file path
            )

            # Prefer server-rendered overlay if your workflow outputs it
            overlay_b64 = get_output(result, "output_image")
            if overlay_b64:
                arr = np.frombuffer(base64.b64decode(overlay_b64), np.uint8)
                display = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                pred_container = get_output(result, "detection_predictions") or get_output(result, "predictions")
                if not pred_container:
                    display = frame.copy()
                    cv2.putText(display, "No predictions", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                else:
                    display = draw_boxes_bgr(frame.copy(), pred_container["predictions"])

        except Exception as e:
            display = frame.copy()
            cv2.putText(display, f"Error: {str(e)[:60]}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        cv2.imshow("TicTacToe Detector (SDK)", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
