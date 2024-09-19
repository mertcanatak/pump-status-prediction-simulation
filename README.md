This project uses an LSTM model to predict the status of a pump based on sensor data. 
The application is built using Streamlit for visualization and DynamoDB for data storage. 
The project includes generating synthetic data, applying the LSTM model for predictions, and saving the results to a database.

## Dataset
The sensor data used in this project is sourced from Kaggle. You can find the original dataset [here](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data/data).

## Running the Project
To run this project, you need to install the dependencies listed in requirements.txt.
```python
pip install -r requirements.txt
```
#### Set Environment Variables
This project requires AWS credentials for connecting to DynamoDB. You can provide your AWS credentials in a .env file or export them as environment variables.
Create a .env file in the root directory with the following content:
```python
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
REGION=your-region
```
#### Running the Application
To run the Streamlit application, navigate to the project directory and run: 
```python
streamlit run app.py
```
## License

[MIT](https://choosealicense.com/licenses/mit/)
