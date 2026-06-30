# AWS Lambda Python base image (container-image Lambda supports large ML deps up to 10 GB).
# --platform=linux/amd64 forces an x86_64 image even when building on an Apple Silicon
# (arm64) Mac, so it matches the function's x86_64 architecture in template.yaml.
FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

# LAMBDA_TASK_ROOT is set by the base image (/var/task)
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt

# Application code and model
COPY handler.py ${LAMBDA_TASK_ROOT}/
COPY model/ ${LAMBDA_TASK_ROOT}/model/

# Handler entrypoint: <file>.<function>
CMD [ "handler.lambda_handler" ]
