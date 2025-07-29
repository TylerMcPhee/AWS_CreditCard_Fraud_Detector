import boto3
import json
import ast
#change default timeout to at least 1 min
def lambda_handler(event, context):
    runtime_client = boto3.client('runtime.sagemaker')
    endpoint_name = 'xgboost-2025-16-25-17-22-29-589'

    # Parse input JSON body 
    input_data = ast.literal_eval(event['body'])

    features = [str(input_data[f'x{i}']) for i in range(1, 32)] 

    # Create CSV string for inference
    sample = ','.join(features)

    # Call SageMaker endpoint
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=sample
    )

    # Read and parse prediction result
    result = int(float(response['Body'].read().decode('utf-8')))

    return {
        'statusCode': 200,
        'body': json.dumps({'prediction': result})
    }