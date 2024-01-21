import boto3

def read_database_file(bucket_name, file_name):
    s3 = boto3.client('s3')

    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_name)
        contents = response['Body'].read().decode('utf-8')  # read the file contents
        return contents
    except Exception as e:
        print(f"Error reading file: {str(e)}")


# Example usage
bucket_name = 'your-bucket-name'
file_name = 'your-database-file.db'

database_contents = read_database_file(bucket_name, file_name)
print(database_contents)


# Access to files stored on AWS S3 bucket
# Create a session using your AWS access and secret keys
session = boto3.Session(
    aws_access_key_id='YOUR_ACCESS_KEY',
    aws_secret_access_key='YOUR_SECRET_KEY'
)

# Create an S3 client using the session
s3_client = session.client('s3')

# Specify the bucket name and file key
bucket_name = 'your-bucket-name'
file_name = 'path/to/your-file'

try:
    # Retrieve the file from the S3 bucket
    response = s3_client.get_object(Bucket=bucket_name, Key=file_name)

    # Read the file content
    file_content = response['Body'].read().decode('utf-8')

    # Print the file content
    print(file_content)

except Exception as e:
    print(f'Error accessing file: {str(e)}')

