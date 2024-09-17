import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
import os

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
AWS_BUCKET = os.getenv('AWS_BUCKET')

file_name = 'models/best_model.keras'
s3_file_name = 'best_model.keras'

def upload_to_s3(file_name, bucket, s3_file_name):
    try:
        s3 = boto3.client('s3',
                          region_name=AWS_DEFAULT_REGION,
                          aws_access_key_id=AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                          config=boto3.session.Config(signature_version='s3v4', s3={'addressing_style': 'path'})
                         )

        s3.upload_file(file_name, bucket, s3_file_name)
        print(f"Upload Successful: {s3_file_name} to bucket {bucket}")
    except FileNotFoundError:
        print(f"The file {file_name} was not found")
    except NoCredentialsError:
        print("Credentials not available")

if __name__ == "__main__":
    upload_to_s3(file_name, AWS_BUCKET, s3_file_name)
