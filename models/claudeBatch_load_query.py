import time
import boto3
import json
import os

BEDROCK_REGION = os.getenv("MEMT_BEDROCK_REGION", "us-east-2")
BEDROCK_ROLE_ARN = os.getenv("MEMT_BEDROCK_ROLE_ARN")

class ClaudeBatchM:
    """
    Lightweight wrapper for Bedrock batch inference:
      1) prepare_input_config → build and upload batch inputs
      2) submit_and_wait → submit a job and wait until completion
    """

    def __init__(self, model: str):
        # Use standard boto3 credentials; optional role assumption is configured
        # through environment variables and is intentionally not hard-coded.
        session = boto3.Session(region_name=BEDROCK_REGION)
        if BEDROCK_ROLE_ARN:
            credentials = session.client('sts').assume_role(
                RoleArn=BEDROCK_ROLE_ARN,
                RoleSessionName="memt-session",
                DurationSeconds=43200,
            )["Credentials"]
            session = boto3.Session(
                region_name=BEDROCK_REGION,
                aws_access_key_id=credentials["AccessKeyId"],
                aws_secret_access_key=credentials["SecretAccessKey"],
                aws_session_token=credentials["SessionToken"],
            )
        self.bedrock = session.client("bedrock")
        self.s3 = session.client("s3")

        self.model = model

    # -------------------------
    # utils (self-contained)
    # -------------------------
    @staticmethod
    def _write_jsonl_local(filepath: str, rows: list):
        with open(filepath, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _upload_file_to_s3(self, local_path: str, bucket: str, key: str):
        self.s3.upload_file(local_path, bucket, key)

    # -------------------------
    # main methods
    # -------------------------
    def prepare_input_config(
        self,
        system_prompts,
        user_prompts,
        local_input_file: str,
        s3_bucket: str,
        s3_key:str,
        max_tokens: int = 1000,
        temperature: float = 0,
        json_schema: dict = None,
        tool_name: str = "structured_output",
        tool_description: str = "Return the answer strictly as JSON matching the input_schema.",
    ):
        """
        Build JSONL input for batch inference and upload to S3
        """
        items = []
        for i, (sys_p, usr_p) in enumerate(zip(system_prompts, user_prompts)):
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": sys_p,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": usr_p}]}
                ],
            }
            if json_schema is not None:
                body["tools"] = [{
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": json_schema
                }]
                body["tool_choice"] = {"type": "tool", "name": tool_name}
            items.append({"recordId": i, "modelInput": body})

        with open(local_input_file, "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        self.s3.upload_file(local_input_file, s3_bucket, s3_key)

    def submit_and_wait(
        self,
        job_name: str,
        s3_bucket: str,
        s3_input_dir: str,
        s3_input_file: str,
        s3_output_dir: str,
        local_output_file:str
    ) -> dict:
        """
        Submit a batch job and wait until it finishes.
        """
        
        resp = self.bedrock.create_model_invocation_job(
            roleArn=BEDROCK_ROLE_ARN or os.environ["MEMT_BEDROCK_ROLE_ARN"],
            modelId=self.model,
            jobName=f"{job_name}-{int(time.time())}",
            inputDataConfig={"s3InputDataConfig": {"s3Uri": f"s3://{s3_bucket}/{s3_input_dir}/{s3_input_file}"}},
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{s3_bucket}/{s3_output_dir}/"}},
        )
        job_arn = resp["jobArn"]
        print(f"submitted job: {job_arn}")

        status_prev=""
        print('processing job...')
        while True:
            response = self.bedrock.get_model_invocation_job(jobIdentifier=job_arn)
            status = response['status']
            if status_prev != status:
                print(status)
                status_prev = status
            if status == 'Completed':
                print("Job succeeded")
                break
            if status in ['Failed', 'Cancelled']:
                print(response)
                raise RuntimeError(f"Job failed with status {status}")
            time.sleep(60)

        suffix_job = job_arn.split('model-invocation-job/')[-1]
        self.s3.download_file(s3_bucket, 
                              f"{s3_output_dir}/{suffix_job}/{s3_input_file}.out", 
                              local_output_file)
        print(f"ouput downloaded to {local_output_file}")

        return job_arn
