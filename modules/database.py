import boto3
from datetime import datetime
from decimal import Decimal
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region = os.getenv('REGION')


dynamodb = boto3.resource(
    'dynamodb',
    region_name=region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

table = dynamodb.Table('Simulated_Data')
table2 = dynamodb.Table('Simulated_Preds')

def insert_data(data):
    try:
        data['datetime'] = data['datetime'].apply(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)
        records = data.to_dict('records')
        for record in records:
            for key, value in record.items():
                if key != 'id' and key != 'datetime':
                    record[key] = Decimal(value)
            response = table.put_item(Item=record)
            print("Veri başarıyla kaydedildi:", response)
    except Exception as e:
        print(f"Veri ekleme hatası: {e}")

def load_data():
    try:
        response = table2.scan()
        data = response['Items']
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return pd.DataFrame()

def save_predictions(record_id, binary_prediction):
    try:
        item = {
            'id': record_id,
            'datetime': datetime.now().isoformat(),
            'prediction': str(binary_prediction)
        }
        response = table2.put_item(Item=item)
        print("Tahmin başarıyla kaydedildi:", response)
    except Exception as e:
        print(f"Tahmin kaydetme hatası: {e}")

def get_last_id():
    try:
        response = table.scan(
            ProjectionExpression='id'
        )
        items = response.get('Items', [])

        if items:
            id_list = [int(item['id']) for item in items]
            return max(id_list)
        else:
            return None
    except Exception as e:
        print(f"Son ID'yi alma hatası: {e}")
        return None


