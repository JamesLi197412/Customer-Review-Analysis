import pandas as pd
import openpyxl
from io import BytesIO
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.workbook import Workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Alignment
from pathlib import Path
from excel_graph import ExcelChart

# Import the necessary libraries for interacting with OneDrive or SharePoint (via Microsoft Graph API)
from msal import PublicClientApplication
from microsoftgraph.client import Client

# Credentials for accessing the Microsoft Graph API
CLIENT_ID = "<Your Client ID>"
CLIENT_SECRET = "<Your Client Secret>"
TENANT_ID = "<Your Tenant ID>"
SCOPES = ['https://graph.microsoft.com/.default']


def export_s3_to_excel(bucket_name, file_key, excel_file_name):
    # Load the file from S3 bucket
    data = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(data['Body'])  # Assuming the file is a CSV, change accordingly if it's a different format

    # Create a new Excel workbook
    wb = Workbook()

    # Add the DataFrame to a new sheet
    ws = wb.active
    for row in dataframe_to_rows(df, index=False, header=True):
        ws.append(row)

    # Create a chart using the ExcelChart class (optional)
    chart = ExcelChart(ws)
    chart.build()

    # Save the spreadsheet to a BytesIO buffer
    excel_file_buffer = BytesIO()
    wb.save(excel_file_buffer)
    excel_file_buffer.seek(0)

    # Authenticate with Microsoft Graph API
    app = PublicClientApplication(CLIENT_ID, authority=f'https://login.microsoftonline.com/{TENANT_ID}')
    token_response = app.acquire_token_for_client(scopes=SCOPES)
    access_token = token_response['access_token']

    # Create a Microsoft Graph API client
    client = Client(CLIENT_ID, CLIENT_SECRET, token=access_token)

    # Upload the Excel file to OneDrive or SharePoint
    response = client.api('/me/drive/root:/path/to/folder/').children[excel_file_name].content.request().put(
        excel_file_buffer)

    print('Excel file uploaded successfully!')


# Usage example
bucket_name = "<Your S3 Bucket Name>"
file_key = "<Your S3 File Key>"
excel_file_name = "<Desired Excel File Name>.xlsx"

export_s3_to_excel(bucket_name, file_key, excel_file_name)
