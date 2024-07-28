import json
import logging
import os
from datetime import datetime
from decimal import Decimal

import boto3
import pymysql
from botocore.exceptions import ClientError
from faker import Faker

# Constants
SUCCESS = "SUCCESS"
FAILED = "FAILED"
REGION_NAME = "us-east-1"

# Initialize Faker for generating fake data (if needed)
fake = Faker()

# Logger setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Secrets Manager details
secret_name = os.environ["secret_arn"]


def get_conn():
    """
    Establishes a connection to the RDS database using credentials from Secrets Manager.

    Returns:
        pymysql.connections.Connection: The database connection object.
    """
    secret = get_secret()
    secret_data = json.loads(secret)

    host = secret_data["host"]
    port = secret_data["port"]
    username = secret_data["username"]
    password = secret_data["password"]
    db_name = secret_data["dbname"]

    conn = pymysql.connect(
        host=host, user=username, passwd=password, db=db_name, connect_timeout=5
    )
    return conn


def get_secret():
    """
    Retrieves the database credentials from AWS Secrets Manager.

    Returns:
        str: The secret string containing the database credentials.
    """
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=REGION_NAME)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        return get_secret_value_response["SecretString"]
    except ClientError as e:
        logger.error("Failed to retrieve secrets: %s", str(e))
        raise e


def return_customer_info(cust_name):
    """
    Retrieves customer information from the CustomerInfo table based on the provided customer name.

    Args:
        cust_name (str): The name of the customer to search for.

    Returns:
        dict: A dictionary containing customer information.
    """
    conn = get_conn()
    with conn.cursor() as cursor:
        query = "SELECT customerId, customerName, Addr1, Addr2, City, State, Zipcode, PreferredActivity, ShoeSize, OtherInfo FROM CustomerInfo WHERE customerName LIKE %s"
        cursor.execute(query, ("%" + cust_name + "%",))
        resp = cursor.fetchall()

        # Add column names to response values
        if resp:
            names = [description[0] for description in cursor.description]
            val_dict = {names[i]: resp[0][i] for i in range(len(names))}
            logger.info("Customer info retrieved")
            return val_dict
        else:
            logger.info("Customer not found")
            return {}


def return_shoe_inventory():
    """
    Retrieves shoe inventory information from the ShoeInventory table.

    Returns:
        list: A list of dictionaries containing the shoe inventory details.
    """
    conn = get_conn()
    with conn.cursor() as cursor:
        query = "SELECT ShoeID, BestFitActivity, StyleDesc, ShoeColors, Price, InvCount FROM ShoeInventory"
        cursor.execute(query)
        resp = cursor.fetchall()

        # Add column names to response values
        names = [description[0] for description in cursor.description]
        val_list = [{name: value for name, value in zip(names, item)} for item in resp]

    logger.info("Shoe inventory retrieved")
    return val_list


def place_shoe_order(shoe_id, cust_id):
    """
    Places a shoe order by updating the ShoeInventory and OrderDetails tables.

    Args:
        shoe_id (int): The ID of the shoe to order.
        cust_id (int): The ID of the customer placing the order.

    Returns:
        int: 1 if the order is placed successfully, 0 otherwise.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cursor:
            # Update ShoeInventory table
            query = "UPDATE ShoeInventory SET InvCount = InvCount - 1 WHERE ShoeID = %s"
            cursor.execute(query, (shoe_id,))

            # Insert order details into OrderDetails table
            today = datetime.today().strftime("%Y-%m-%d")
            query = "INSERT INTO OrderDetails (orderdate, shoeId, CustomerId) VALUES (%s, %s, %s)"
            cursor.execute(query, (today, shoe_id, cust_id))

        conn.commit()
        logger.info("Shoe order placed")
        return 1
    except Exception as e:
        logger.error("Error placing shoe order: %s", str(e))
        conn.rollback()
        return 0


def lambda_handler(event, context):
    """
    Lambda function handler to process API requests.

    Args:
        event (dict): Event data from API Gateway.
        context (object): Runtime information from Lambda.

    Returns:
        dict: API response.
    """
    logger.info(json.dumps(event))
    responses = []

    api_path = event.get("apiPath")
    logger.info("API Path: %s", api_path)

    if api_path == "/customer/{CustomerName}":
        cust_name = next(
            (
                param["value"]
                for param in event["parameters"]
                if param["name"] == "CustomerName"
            ),
            "",
        )
        body = return_customer_info(cust_name)
    elif api_path == "/place_order":
        shoe_id = next(
            (
                param["value"]
                for param in event["parameters"]
                if param["name"] == "ShoeID"
            ),
            None,
        )
        cust_id = next(
            (
                param["value"]
                for param in event["parameters"]
                if param["name"] == "CustomerID"
            ),
            None,
        )
        body = place_shoe_order(shoe_id, cust_id)
    elif api_path == "/check_inventory":
        body = return_shoe_inventory()
    else:
        body = {
            "message": "{} is not a valid API path, try another one.".format(api_path)
        }

    response_body = {
        "application/json": {"body": json.dumps(body, default=decimal_default)}
    }

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


def decimal_default(obj):
    """
    Convert Decimal objects to float for JSON serialization.

    Args:
        obj (object): The object to convert.

    Returns:
        float: The converted float value.

    Raises:
        TypeError: If the object is not a Decimal.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError
