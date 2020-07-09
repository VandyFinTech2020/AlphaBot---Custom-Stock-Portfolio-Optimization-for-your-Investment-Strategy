### Required Libraries ###
import json
from botocore.vendored import requests
import boto3
import numpy as np
import pandas as pd

    
def marketCapConverter(row):
    if int(row["marketcap"]) > 200000:
        return "mega"
    elif int(row["marketcap"]) > 10000:
        return "large"
    elif int(row["marketcap"]) > 2000:
        return "mid"
    else:
        return "small"
        
def priceSaleConverter(row):
    if float(row["ps"]) > 2:
        return "High"
    elif float(row["ps"]) > 1:
        return "Fair"
    else:
        return "Low"
        
def sharePriceConverter(row):
    if float(row["close"]) > 5:
        return "Above $5"
    else:
        return "Under $5"
        
def tradeVolumeConverter(row):
    if int(row["volume"]) > 100000:
        return "no"
    else:
        return "yes"
        
def priceBookConverter(row):
    if float(row["pb"]) > 2:
        return "High"
    elif float(row["pb"]) > 1:
        return "Fair"
    else:
        return "Low"


def priceEarningsConverter(row):
    if float(row["pe"]) > 20:
        return "High"
    elif float(row["pe"]) > 10:
        return "Average"
    elif float(row["pe"]) > 0:
        return "Low"
    else:
        return "NM"
### Functionality Helper Functions ###

def validate_data_user(intent_request, userAge, contactInfo, marketCap, tradeVolume, sharePrice, priceEarnings, priceSale, priceBook, timeFrame):
    """
    Validates the data provided by the user.
    """

    if userAge is not None:
        if int(userAge) >= 21:
            return userAge
        else:
            return build_validation_result(
                False,
                "userAge",
                "You should be at least 21 years old to use this service, "
                "please provide a new age."
            )
    print(userAge)            
            
    if contactInfo is not None:
        if len(contactInfo) == 10:
            return contactInfo
        else:
            return build_validation_result(
                False,
                "contactInfo",
                "Please provide a 10 digit number."
            )
            
    if marketCap is not None:
        if marketCap == "small":
            return marketCap
        elif marketCap == "mid":
            return marketCap
        elif marketCap == "large":
            return marketCap
        elif marketCap == "mega":
            return marketCap
        else:
            return build_validation_result(
                False,
                "marketCap",
                "You should pick either small, mid, large, or mega, "
                "please provide a new size."
            )
                
    if tradeVolume is not None:
        if tradeVolume == "yes":
            return tradeVolume
        elif tradeVolume == "no":
            return tradeVolume
        else:
            return build_validation_result(
                False,
                "tradeVolume",
                "Please choose yes or no."
            )
    
    if sharePrice is not None:
        if sharePrice == "low":
            return sharePrice
        elif sharePrice == "high":
            return sharePrice
        else:
            return build_validation_result(
                False,
                "sharePrice",
                "Please select either low or high."
            )
            
    if priceEarnings is not None:
        if priceEarnings == "not meaningful":
            return priceEarnings
        elif priceEarnings == "low":
            return priceEarnings
        elif priceEarnings == "average":
            return priceEarnings
        elif priceEarnings == "high":
            return priceEarnings
        else:
            return build_validation_result(
                False,
                "priceEarnings",
                "Please select Not Meaningful, Low, Average, High."
            )
                
    if priceSale is not None:
        if priceSale == "low":
            return priceSale
        elif priceSale == "fair":
            return priceSale
        elif priceSale == "high":
            return priceSale
        else:
            return build_validation_result(
                False,
                "priceSale",
                "Please select low, fair, or high."
            )
                
    if priceBook is not None:
        if priceBook == "low":
            return priceBook
        elif priceBook == "fair":
            return priceBook
        elif priceBook == "high":
            return priceBook
        else:
            return build_validation_result(
                False,
                "priceBook",
                "Please select low, fair, or high."
            )
                
    if timeFrame is not None:
        if timeFrame >= 1:
            return timeFrame
        elif timeFrame <= 2000:
            return timeFrame
        else:
            return build_validation_result(
                False,
                "timeFrame",
                "Please select a time frame between 1 and 2000 days."
            )

    # A True results is returned if age or marketcap are valid
    return build_validation_result(True, None, None)

def validate_data_okay(intent_request):
    
    return build_validation_result(True, None, None)

def build_validation_result(is_valid, violated_slot, message_content):
    """
    Defines an internal validation message structured as a python dictionary.
    """
    if message_content is None:
        return {"isValid": is_valid, "violatedSlot": violated_slot}

    return {
        "isValid": is_valid,
        "violatedSlot": violated_slot,
        "message": {"contentType": "PlainText", "content": message_content},
    }


### Dialog Actions Helper Functions ###
def get_slots(intent_request):
    """
    Fetch all the slots and their values from the current intent.
    """
    return intent_request["currentIntent"]["slots"]


def elicit_slot(session_attributes, intent_name, slots, slot_to_elicit, message):
    """
    Defines an elicit slot type response.
    """

    return {
        "sessionAttributes": session_attributes,
        "dialogAction": {
            "type": "ElicitSlot",
            "intentName": intent_name,
            "slots": slots,
            "slotToElicit": slot_to_elicit,
            "message": message,
        },
    }


def delegate(session_attributes, slots):
    """
    Defines a delegate slot type response.
    """

    return {
        "sessionAttributes": session_attributes,
        "dialogAction": {"type": "Delegate", "slots": slots},
    }


def close(session_attributes, fulfillment_state, message):
    """
    Defines a close slot type response.
    """

    response = {
        "sessionAttributes": session_attributes,
        "dialogAction": {
            "type": "Close",
            "fulfillmentState": fulfillment_state,
            "message": message,
        },
    }

    return response

def MLIntent(intent_request):
    """
    Performs dialog management and fulfillment for recommending a portfolio.
    """
    # tickers = get_symbols(api)
    # prices_df = get_prices(tickers) # same as data csv
    # print(get_slots(intent_request))
    
    # Gets slots' values
    firstName = get_slots(intent_request)["firstName"]
    userAge = get_slots(intent_request)["userAge"]
    contactInfo = get_slots(intent_request)["contactInfo"]
    marketCap = get_slots(intent_request)["marketCap"]
    tradeVolume = get_slots(intent_request)["tradeVolume"]
    sharePrice = get_slots(intent_request)["sharePrice"]
    priceEarnings = get_slots(intent_request)["priceEarnings"]
    priceSale = get_slots(intent_request)["priceSale"]
    priceBook = get_slots(intent_request)["priceBook"]
    timeFrame = get_slots(intent_request)["timeFrame"]

    # Gets the invocation source, for Lex dialogs "DialogCodeHook" is expected.
    source = intent_request["invocationSource"]  

    if source == "DialogCodeHook":
        # This code performs basic validation on the supplied input slots.

        # Gets all the slots
        slots = get_slots(intent_request)
        
        # Validates user's input using the validate_data function
        validation_result = validate_data_okay(intent_request)

        # If the data provided by the user is not valid,
        # the elicitSlot dialog action is used to re-prompt for the first violation detected.
        if not validation_result["isValid"]:
            slots[validation_result["violatedSlot"]
                  ] = None  # Cleans invalid slot

            # Returns an elicitSlot dialog to request new data for the invalid slot
            return elicit_slot(
                intent_request["sessionAttributes"],
                intent_request["currentIntent"]["name"],
                slots,
                validation_result["violatedSlot"],
                validation_result["message"],
            )

        # Fetch current session attributes
        output_session_attributes = intent_request["sessionAttributes"]

        # Once all slots are valid, a delegate dialog is returned to Lex to choose the next course of action.
        return delegate(output_session_attributes, get_slots(intent_request))
        
    # take valid values from slots, convert to variables, use variables to filter dataframe  
    s3_client = boto3.client('s3')
    file = s3_client.get_object(Bucket="alphabot", Key="fullprocess.csv")
    prices_df = pd.read_csv(file["Body"])
    print(prices_df.head())
    prices_df["marketcapname"] = prices_df.apply(marketCapConverter, axis=1)
    prices_df["volumename"] = prices_df.apply(tradeVolumeConverter, axis=1)
    prices_df["pricebookname"] = prices_df.apply(priceBookConverter, axis=1)
    prices_df["priceearningsname"] = prices_df.apply(priceEarningsConverter, axis=1)
    prices_df["pricesalename"] = prices_df.apply(priceSaleConverter, axis=1)
    prices_df["sharepricename"] = prices_df.apply(sharePriceConverter, axis=1)
    
    mCR_df = prices_df[prices_df["marketcapname"] == marketCap]
    print(mCR_df.shape)
    pER_df = ""
    stocks = False
    if not mCR_df.empty:     
        tVR_df = prices_df[prices_df["volumename"] == tradeVolume]
        print(tVR_df.shape)
        if not tVR_df.empty:
            sPR_df = tVR_df[tVR_df["sharepricename"] == sharePrice]
            print("close")
            if not sPR_df.empty:   
                pBR_df = sPR_df[sPR_df["pricebookname"] == priceBook]
                print("pb")
                if not pBR_df.empty:    
                    pSR_df = pBR_df[pBR_df["pricesalename"] == priceSale]
                    print("ps")
                    print(pSR_df)
                    if not pSR_df.empty:   
                        pER_df = pSR_df[pSR_df["priceearningsname"] == priceEarnings]
                        stocks = True
                        
                        
    
    intent_request["sessionAttributes"] = {"marketCap": marketCap, 
        "tradeVolume": tradeVolume,
        "sharePrice": sharePrice, "priceEarnings": priceEarnings, 
        "priceSale": priceSale, "priceBook": priceBook, "timeFrame": timeFrame, "final_df": pER_df}
    
    if stocks:
        message = "Cool we have stocks"
    else:
        message = "We filtered out all the stocks"
# Return a message with conversion's result.
    return close(
        intent_request["sessionAttributes"],
        "Fulfilled",
        {
            "contentType": "PlainText",
            "content": message
        }
    )

### Intents Dispatcher ###
def dispatch(intent_request):
    """
    Called when the user specifies an intent for this bot.
    """

    # Get the name of the current intent
    intent_name = intent_request["currentIntent"]["name"]

    # Dispatch to bot's intent handlers
    if intent_name == "MLIntent":
        return MLIntent(intent_request)

    raise Exception("Intent with name " + intent_name + " not supported")


### Main Handler ###
def lambda_handler(event, context):
    """
    Route the incoming request based on intent.
    The JSON body of the request is provided in the event slot.
    """

    return dispatch(event)

