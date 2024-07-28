import json
import logging
import os
import sqlite3
from datetime import datetime

import boto3

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client and set up environment variables
s3 = boto3.client("s3")
bucket = os.environ.get("BUCKET_NAME")  # Name of bucket with data file
db_name = "demo_csbot_db"  # Name of the database file in S3
local_db = "/tmp/csbot.db"  # Local path in Lambda /tmp folder for the database file

# Download the database file from S3 to the local /tmp folder
s3.download_file(bucket, db_name, local_db)

# Initialize SQLite3 connection and cursor
conn = None
cursor = None


def load_data():
    """Load the SQLite database from the local file and create a cursor."""
    global conn, cursor
    conn = sqlite3.connect(local_db)
    cursor = conn.cursor()
    logger.info("Completed initial data load")
    return cursor


def return_customer_info(custName):
    """
    Retrieve customer information based on customer name.

    Args:
        custName (str): The name of the customer to search for.

    Returns:
        dict: A dictionary containing customer information.
    """
    query = "SELECT customerId, customerName, Addr1, Addr2, City, State, Zipcode, PreferredActivity, ShoeSize, OtherInfo FROM CustomerInfo WHERE customerName LIKE ?"
    cursor.execute(query, ("%" + custName + "%",))
    resp = cursor.fetchall()

    # Add column names to response values
    if resp:
        names = [description[0] for description in cursor.description]
        valDict = {names[i]: resp[0][i] for i in range(len(names))}
        logger.info("Customer info retrieved")
        return valDict
    else:
        logger.info("Customer not found")
        return {}


def return_shoe_inventory():
    """
    Retrieve shoe inventory information.

    Returns:
        list: A list of dictionaries containing shoe inventory details.
    """
    query = "SELECT ShoeID, BestFitActivity, StyleDesc, ShoeColors, Price, InvCount FROM ShoeInventory"
    cursor.execute(query)
    resp = cursor.fetchall()

    # Add column names to response values
    names = [description[0] for description in cursor.description]
    valDict = [{names[i]: item[i] for i in range(len(names))} for item in resp]
    logger.info("Shoe inventory retrieved")
    return valDict


def place_shoe_order(ssId, custId):
    """
    Place a shoe order by reducing inventory and updating the order details table.

    Args:
        ssId (int): Shoe ID to be ordered.
        custId (int): Customer ID placing the order.

    Returns:
        int: Status code (1 for success).
    """
    global cursor, conn
    try:
        # Update inventory count
        query = "UPDATE ShoeInventory SET InvCount = InvCount - 1 WHERE ShoeID = ?"
        cursor.execute(query, (ssId,))

        # Insert order details
        today = datetime.today().strftime("%Y-%m-%d")
        query = (
            "INSERT INTO OrderDetails (orderdate, shoeId, CustomerId) VALUES (?, ?, ?)"
        )
        cursor.execute(query, (today, ssId, custId))
        conn.commit()

        # Upload updated database file to S3
        s3.upload_file(local_db, bucket, db_name)
        cursor = None  # Force reload of data
        logger.info("Shoe order placed")
        return 1
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return 0


def lambda_handler(event, context):
    """
    AWS Lambda handler function to process API requests.

    Args:
        event (dict): Event data from API Gateway.
        context (object): Runtime information from Lambda.

    Returns:
        dict: API response.
    """
    global cursor
    responses = []

    if cursor is None:
        cursor = load_data()

    api_path = event.get("apiPath")
    logger.info(f"API Path: {api_path}")

    parameters = event.get("parameters", [])
    body = {}

    if api_path == "/customer/{CustomerName}":
        custName = next(
            (param["value"] for param in parameters if param["name"] == "CustomerName"),
            "",
        )
        body = return_customer_info(custName)
    elif api_path == "/place_order":
        ssId = next(
            (param["value"] for param in parameters if param["name"] == "ShoeID"), ""
        )
        custId = next(
            (param["value"] for param in parameters if param["name"] == "CustomerID"),
            "",
        )
        body = place_shoe_order(ssId, custId)
    elif api_path == "/check_inventory":
        body = return_shoe_inventory()
    else:
        body = {"message": f"{api_path} is not a valid API, try another one."}

    response_body = {"application/json": {"body": json.dumps(body)}}

    action_response = {
        "actionGroup": event.get("actionGroup"),
        "apiPath": api_path,
        "httpMethod": event.get("httpMethod"),
        "httpStatusCode": 200,
        "responseBody": response_body,
    }

    responses.append(action_response)

    api_response = {"messageVersion": "1.0", "response": action_response}

    return api_response
