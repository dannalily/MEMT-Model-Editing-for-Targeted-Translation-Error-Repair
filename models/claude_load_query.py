import json
import os
import boto3
from typing import Optional, Tuple, Any, Dict
BEDROCK_REGION = os.getenv("MEMT_BEDROCK_REGION", "us-east-1")
BEDROCK_ROLE_ARN = os.getenv("MEMT_BEDROCK_ROLE_ARN")

class ClaudeM:
    def __init__(self, model):
        # Use the standard boto3 credential chain. Set MEMT_BEDROCK_ROLE_ARN
        # only when an optional cross-account role is required.
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
        self.bedrock = session.client("bedrock-runtime")
        self.model = model

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0,
        json_schema: Optional[Dict[str, Any]] = None,
        tool_name: str = "structured_output",
        tool_description: str = "Return the answer strictly as JSON matching the input_schema."
    ) -> Tuple[str, Dict[str, Any]]:
        """
        If `json_schema` is provided, enforce structured output via Anthropic tool use.
        Returns (response_text, full_response_dict). When structured output is used,
        `response_text` will be the JSON string from the tool call and
        full_response_dict['structured_output'] will contain the parsed object.
        """
        user_message = [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]

        body: Dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": user_message
        }

        # If a schema is provided, require structured output using tool choice.
        if json_schema is not None:
            body["tools"] = [{
                "name": tool_name,
                "description": tool_description,
                "input_schema": json_schema  # must be a valid JSON Schema
            }]
            # Force the model to call THIS tool so output conforms to the schema.
            body["tool_choice"] = {"type": "tool", "name": tool_name}

        inp = {
            "modelId": self.model,
            "contentType": "application/json",
            "accept": "application/json",
            "body": json.dumps(body)
        }

        response = self.bedrock.invoke_model(**inp)
        response_body = json.loads(response.get('body').read())

        # Default behavior: plain text in the first text block.
        response_text = None
        structured_obj = None

        # If tool use was enforced, Claude will respond with a tool_use block.
        # Extract the tool input (already structured) and also keep a JSON string version.
        if json_schema is not None:
            for block in response_body.get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == tool_name:
                    tool_input = block.get("input")
                    structured_obj = tool_input
                    response_text = json.dumps(tool_input, ensure_ascii=False)
                    break

            # Fallback: if no tool_use was returned (should be rare), try text.
            if response_text is None:
                # try to get text content and pass it back raw
                for block in response_body.get("content", []):
                    if block.get("type") == "text":
                        response_text = block.get("text")
                        break
        else:
            # No schema path (original behavior)
            # Anthropic messages return a list of content blocks; first text is typical
            for block in response_body.get("content", []):
                if block.get("type") == "text":
                    response_text = block.get("text")
                    break

        # Guarantee a string return
        if response_text is None:
            response_text = ""

        # Attach parsed structured output (if any) to the response dict for convenience
        if structured_obj is not None:
            response_body["structured_output"] = structured_obj

        return response_text, response_body
    
