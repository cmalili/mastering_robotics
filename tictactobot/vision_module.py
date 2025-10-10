import cv2, os, tempfile, base64, numpy as np
import json
from inference_sdk import InferenceHTTPClient

API_URL   = "http://localhost:9001"
WORKSPACE = "chris-hub"
WORKFLOW  = "detect-and-classify-3"
API_KEY   = "1HhSNS3VWex8YfHgeGzJ"

client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)
tmp_path = os.path.join(tempfile.gettempdir(), "rf_frame.jpg")

# keep one persistent camera handle
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam")
print("[Vision] Camera initialized once.")

def draw_boxes_bgr(img_bgr, preds):
    for p in preds:
        cx, cy, w, h = p["x"], p["y"], p["width"], p["height"]
        x1, y1 = int(cx - w/2), int(cy - h/2)
        x2, y2 = int(cx + w/2), int(cy + h/2)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_bgr,
            f'{p["class"]}:{p["confidence"]:.2f}',
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return img_bgr


def detect_board_state(show=True):
    """Capture one frame and return detections + optional display."""
    ret, frame = cap.read()

    #cap.release()
    if not ret:
        raise RuntimeError("Could not read from webcam")

    cv2.imwrite(tmp_path, frame)

    result = client.run_workflow(
        workspace_name=WORKSPACE,
        workflow_id=WORKFLOW,
        images={"image": tmp_path},
    )

    if isinstance(result, str):
        try:
            result = json.loads(result)
            print("[Vision] Parsed JSON result.")
        except json.JSONDecodeError:
            print("⚠️ Could not parse JSON:", result[:200])
            return []
        
    preds = []

    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    if isinstance(result, dict):
        for key in ("detection_predictions", "predictions"):
            container = result.get(key) or result.get("results", {}).get(key)
            if container:
            # some containers store predictions directly
                if isinstance(container, dict) and "predictions" in container:
                    preds = container["predictions"]
                elif isinstance(container, list):
                    preds = container
                break
    else:
        print("Unexpected result format:", type(result), result)

    if show:
        display = draw_boxes_bgr(frame.copy(), preds)
        cv2.imshow("Vision", display)
        cv2.waitKey(1)
        #cv2.destroyAllWindows()

    return display, preds


def to_grid(preds, width=640, height=480):
    """Map detection coordinates to 3x3 grid."""
    board = [["" for _ in range(3)] for _ in range(3)]
    cell_w, cell_h = width / 3, height / 3

    for p in preds:
        cls = p["class"]
        x, y = p["x"], p["y"]
        c = min(2, int(x // cell_w))
        r = min(2, int(y // cell_h))
        board[r][c] = cls
    return board