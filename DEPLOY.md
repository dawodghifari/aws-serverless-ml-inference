# Deploy guide

Step-by-step to get the landmine-detection API live on your own AWS account. Budget
~45–60 minutes the first time. Light testing stays within the AWS Free Tier.

## 0. Prerequisites (one-time)

1. **An AWS account** — https://aws.amazon.com (needs a card; this usage is free-tier).
2. **AWS CLI** — `aws --version` (https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
3. **AWS SAM CLI** — `sam --version` (https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
4. **Docker** — `docker --version` (must be running; SAM builds the Lambda image with it)
5. **Python 3.12** locally, plus `pip install ultralytics onnx onnxruntime` for the export step.

Configure credentials once:

```bash
aws configure          # paste your Access Key ID, Secret, region (e.g. ap-southeast-2)
```

> Security note: never commit credentials. `aws configure` stores them in `~/.aws/`,
> outside this repo, and `.gitignore` already excludes the model + build artefacts.

## 1. Export the model to ONNX

Use **your** trained weights (the `best.pt` from the landmine training notebook):

```bash
cd aws-serverless-ml-inference
python scripts/export_model.py --weights /path/to/best.pt --names landmine
# -> app/model/model.onnx  and  app/model/labels.json
```

Don't have the weights handy yet? Deploy a pretrained COCO model first to prove the
pipeline end-to-end, then swap in your own later:

```bash
python scripts/export_model.py --pretrained yolo11n.pt
```

> If you trained at a non-default image size, pass `--imgsz <N>` here and set
> `INPUT_SIZE` in `app/handler.py` to the same value.

## 2. Build

```bash
sam build
```

SAM reads `template.yaml`, builds the container image from `app/Dockerfile`, and bundles
ONNX Runtime + your model.

## 3. Deploy

```bash
sam deploy --guided
```

Answer the prompts:
- Stack name: `landmine-detection-api`
- AWS Region: your region (e.g. `ap-southeast-2`)
- Confirm changes before deploy: `y`
- Allow SAM to create IAM roles: `y`
- Create an ECR repo for the image: `y`

When it finishes, copy **ApiBaseUrl** / **PredictEndpoint** from the Outputs.

## 4. Test it

```bash
# any thermal test image
python scripts/test_request.py https://<api-id>.execute-api.<region>.amazonaws.com test_thermal.jpg
```

You'll get the detections printed and an `annotated.png` saved locally. Then open
**`web/index.html`** in a browser, paste your `ApiBaseUrl`, and drop an image to see the
boxes drawn live. Check the logs endpoint too:

```bash
curl https://<api-id>.execute-api.<region>.amazonaws.com/stats
```

## 5. Tear down (avoid any charges)

```bash
sam delete --stack-name landmine-detection-api
```

This removes the Lambda, API Gateway, DynamoDB table, and ECR image.

---

## Troubleshooting

- **`sam build` fails on Docker** — make sure Docker Desktop is running.
- **Slow first call** — expected on first invoke (cold start loads the model); subsequent
  calls are fast. Memory is set to 3008 MB in `template.yaml` (more memory = more vCPU).
- **No detections / too many** — adjust `conf` in the request body (lower = more recall,
  higher = more precision).
- **403 / auth errors** — re-run `aws configure`; check the region matches your deploy.
- **Boxes look offset** — `INPUT_SIZE` in `handler.py` must match the `--imgsz` you exported with.

## What to put on your CV after this is deployed and working

> **Serverless Landmine-Detection API (AWS).** Deployed a YOLO11 object-detection model as a
> serverless inference service: exported to **ONNX**, packaged as a container-image **AWS
> Lambda** behind **API Gateway**, with NumPy letterbox/NMS post-processing and every
> inference logged to **DynamoDB**; infrastructure defined as code with **AWS SAM**, plus a
> browser demo that renders detections live.

Add the GitHub repo link, and you can truthfully list **AWS (Lambda, API Gateway, DynamoDB,
SAM), ONNX Runtime, Docker, serverless, IaC, computer vision** in your skills.

## Interview talking points (Amazon Leadership Principles)

- **Ownership / Deliver Results:** you took a model all the way to a live, callable API with a demo.
- **Dive Deep:** container-image Lambda to fit ML deps; why letterbox + NMS in NumPy rather
  than pulling in torch; cold-start vs latency vs memory trade-offs; scan now vs GSI at scale.
- **Learn and Be Curious:** you taught yourself the AWS serverless stack and IaC to ship it.
