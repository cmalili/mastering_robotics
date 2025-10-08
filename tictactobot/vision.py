from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001", # use local inference server
    api_key="1HhSNS3VWex8YfHgeGzJ"
)

result = client.run_workflow(
    workspace_name="chris-hub",
    workflow_id="detect-and-classify-3",
    images={
        "image": "roboflow_test.png"
    }
)


# --- helper to fetch an output by name from dict OR list-of-dicts ---
def get_output(result, name):
    if isinstance(result, dict):
        return result.get(name)
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and name in item:
                return item[name]
    return None

# ----- after you call client.run_workflow(...) and get `result` -----
pred_container = get_output(result, "detection_predictions") or get_output(result, "predictions")
if pred_container is None:
    raise RuntimeError("No predictions found in workflow result. Check your `outputs` in the workflow.")

boxes = pred_container["predictions"]   # <-- NOW boxes is defined

# Example: draw the boxes on your local image
import cv2
img = cv2.imread("roboflow_test.png")   # same image you sent to the workflow
for p in boxes:
    cx, cy, w, h = p["x"], p["y"], p["width"], p["height"]
    x1, y1 = int(cx - w/2), int(cy - h/2)
    x2, y2 = int(cx + w/2), int(cy + h/2)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img, f'{p["class"]}:{p["confidence"]:.2f}',
                (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

cv2.imwrite("overlay_manual.png", img)
print("Saved overlay_manual.png")
