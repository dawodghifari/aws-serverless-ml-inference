"""
Send a local image to the deployed detection API, print the detections, and save
the annotated image returned by the service.

Usage:
    python scripts/test_request.py https://<api-id>.execute-api.<region>.amazonaws.com test.jpg
"""

import base64
import json
import sys
import urllib.request


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/test_request.py <API_BASE_URL> <IMAGE_PATH>")
        sys.exit(1)

    base_url, image_path = sys.argv[1].rstrip("/"), sys.argv[2]
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    body = json.dumps({"image_base64": image_b64, "conf": 0.25, "annotate": True}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/predict", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        result = json.load(resp)

    # Print everything except the (large) base64 image.
    summary = {k: v for k, v in result.items() if k != "annotated_image_base64"}
    print(json.dumps(summary, indent=2))

    if result.get("annotated_image_base64"):
        out = "annotated.png"
        with open(out, "wb") as f:
            f.write(base64.b64decode(result["annotated_image_base64"]))
        print(f"\nAnnotated image saved -> {out}")


if __name__ == "__main__":
    main()
