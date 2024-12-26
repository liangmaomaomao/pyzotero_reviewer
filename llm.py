import boto3
def get_bedrock_response(user_message, model_id='us.meta.llama3-2-11b-instruct-v1:0'):
    # Create a Bedrock Runtime client in the AWS Region you want to use.
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    conversation = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]
    if 'llama' in model_id:
        response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens":2048,"temperature":0.5,"topP":0.9},
        additionalModelRequestFields={}
        )

    if 'mistral' in conversation:
        response = client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens":400,"temperature":0.4,"topP":0.7},
            additionalModelRequestFields={"top_k":50}
        )

    # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]
    return response_text


def template(input_pdf,mode):
    if mode=='conclusion/method':
        prompt = f"""Please read the following academic paper text carefully and summarize its main conclusions and method they applied, identify the main country or region studied, as well as the time period of the research. The output should be concise and comprehensive.

            Paper:
            {input_pdf}

            Return in this json format:
            {{"Conclusion":"Paper's Conclusion","Method":"Paper's Method to research","region":"Paper studies region","Time Period":"Paper's time period"}}
            """
    if mode=='time/region':
        prompt = f"""Please read the following academic paper text carefully and identify the main country or region studied, as well as the time period of the research. The output should be concise.

            Paper:
            {input_pdf}

            Return in this format:
            {{region:'Paper studies region',Time Period:'Paper's time period'}}
            """
    return prompt
    
