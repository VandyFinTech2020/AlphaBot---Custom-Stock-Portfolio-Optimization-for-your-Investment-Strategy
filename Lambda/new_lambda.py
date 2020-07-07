### Required Libraries ###
import json
from botocore.vendored import requests

import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from time import sleep
import quandl
import os
quandl.ApiConfig.api_key = os.getenv("QUANDL_API_KEY")
alpaca_api_key = os.getenv("ALPACA_API_KEY")
alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, api_version='v2')
### Functionality Helper Functions ###
def get_symbols(api):
    """
    Return an updated list of the symbols of the tradeable assets
    """
    assets = api.list_assets()
    tradeable = [asset for asset in assets if asset.tradable ]
    symbols = [asset.symbol for asset in tradeable]
    return(symbols)
    
def get_prices(symbols):
    """
    Return an updated list of the symbols of the tradeable assets
    """
    timeframe = '1D'
    start_date = pd.Timestamp("2018-06-18", tz="America/New_York").isoformat()
    end_date = pd.Timestamp("2020-06-23", tz="America/New_York").isoformat()
    stockprices = []
    i = 0
    for asset in symbols:
        print(asset)
        df = api.get_barset(
            asset,
            timeframe,
            limit=None,
            start=start_date,
            end=end_date,
            after=None,
            until=None,
        ).df
        dfq = quandl.get_table('SHARADAR/DAILY', ticker=asset)
        #print(df)
        if df.empty:
            print("     - Empty - Skip")
        elif dfq.empty:
            print("     - Empty - Skip")
        else:
            # format alpaca df for joining
            df = df.stack(level=0)
            df = df.rename_axis(('date', 'ticker'))
            df = df.reset_index()
            df['date'] = df['date'].dt.date
            df = df.set_index(["date", "ticker"])
            dfq = dfq.set_index(["date", "ticker"])
            dfb = df.join(dfq)
            print(dfb.head(2))
            #stockprices.append({"Ticker" : asset, "Metrics": dfb})
            stockprices.append(dfb)
            # print("Length:", len(stockprices))
            i += 1
            if i%3 == 0:
                sleep(1)
    dfstockprices = pd.concat(stockprices)
    return dfstockprices
    
def trading_indicators(pER_df):
    """Generates trading signals for a given dataset."""
    # Grab just the `date` and `close` from the dataset
    signals_df = data_df.loc[:, ["date", "close"]].copy()

    # Set the `date` column as the index
    signals_df = signals_df.set_index("date", drop=True)

    # Set the short window and long windows
    short_window = 50
    long_window = 200

    # Generate the short and long moving averages (50 and 200 days, respectively)
    signals_df["SMA50"] = signals_df["close"].rolling(window=short_window).mean()
    signals_df["SMA200"] = signals_df["close"].rolling(window=long_window).mean()
    signals_df["Cross"] = 0.0

    # Generate the trading signal 0 or 1,
    # Death Cross Zone = 0 is when the SMA50 < SMA200
    # Golden Cross Zone = 1 is when the SMA50 > SMA200
    signals_df["Cross"][short_window:] = np.where(
        signals_df["SMA50"][short_window:] > signals_df["SMA200"][short_window:],
        1.0,
        0.0,
    )

    # Calculate the points in time at which a position should be taken, 1 or -1
    signals_df["Entry/Exit"] = signals_df["Cross"].diff()
    #Set Bollinger Band window to 20 (standard lag)
    bollinger_window = 20

# Calculate rolling mean and standard deviation
    signals_df['bollinger_mid_band'] = signals_df['Close'].rolling(window=bollinger_window).mean()
    signals_df['bollinger_std'] = signals_df['Close'].rolling(window=20).std()

# Calculate upper and lowers bands of bollinger band. Range set to 2 standard deviations instead of 1
    signals_df['bollinger_upper_band']  = signals_df['bollinger_mid_band'] + (signals_df['bollinger_std'] * 2)
    signals_df['bollinger_lower_band']  = signals_df['bollinger_mid_band'] - (signals_df['bollinger_std'] * 2)

# Calculate bollinger band trading signal
    signals_df['bollinger_long'] = np.where(signals_df['Close'] < signals_df['bollinger_lower_band'], 1.0, 0.0)
    signals_df['bollinger_short'] = np.where(signals_df['Close'] > signals_df['bollinger_upper_band'], -1.0, 0.0)
    signals_df['bollinger_signal'] = signals_df['bollinger_long'] + signals_df['bollinger_short']
        
    
    return signals_df

### Functionality Helper Functions ###

def validate_data_user(intent_request, userAge, contactInfo, marketCap, tradeVolume, dividendPayer, sharePrice, priceEarning, priceSale, priceBook, timeFrame, portfolioSize):
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
                
    if dividendPayer is not None:
        if dividendPayer == "yes":
            return dividendPayer
        elif dividendPayer == "no":
            return dividendPayer
        else:
            return build_validation_result(
                False,
                "dividendPayer",
                "Please answer yes or no."
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
            
    if priceEarning is not None:
        if priceEarning == "not meaningful":
            return priceEarning
        elif priceEarning == "low":
            return priceEarning
        elif priceEarning == "average":
            return priceEarning
        elif priceEarning == "high":
            return priceEarning
        else:
            return build_validation_result(
                False,
                "priceEarning",
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
            
    if portfolioSize is not None:
        if portfolioSize >= 4:
            return portfolioSize
        elif portfolioSize <=10:
            return portfolioSize
        else:
            return build_validation_result(
                False,
                "portfolioSize",
                "Please select a portfolio size between 4 and 10."
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

### Intents Handlers ###
def userProfile(intent_request):
    """
    Performs dialog management and fulfillment for recommending a portfolio.
    """

    # Gets slots' values
    firstName = get_slots(intent_request)["firstName"]
    userAge = get_slots(intent_request)["userAge"]
    contactInfo = get_slots(intent_request)["contactInfo"]

    # Gets the invocation source, for Lex dialogs "DialogCodeHook" is expected.
    source = intent_request["invocationSource"] 

    if source == "DialogCodeHook":
        # This code performs basic validation on the supplied input slots.

        # Gets all the slots
        slots = get_slots(intent_request)

        # Validates user's input using the validate_data function
        validation_result = validate_data_user(intent_request, userAge)

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
    
    intent_request["sessionAttributes"] = {"firstName": firstName, 
        "userAge": userAge, "contactInfo": contactInfo}

    # Return a message with conversion's result.
    return close(
        intent_request["sessionAttributes"],
        "Fulfilled",
        {
            "contentType": "PlainText",
            "content": "Thank you, now ask to filter through stocks."
        },
    )

def filterIntent(intent_request):
    """
    Performs dialog management and fulfillment for recommending a portfolio.
    """

    # Gets slots' values
    marketCap = get_slots(intent_request)["marketCap"]
    tradeVolume = get_slots(intent_request)["tradeVolume"]
    sharePrice = get_slots(intent_request)["sharePrice"]
    priceEarning = get_slots(intent_request)["priceEarning"]
    priceSale = get_slots(intent_request)["priceSale"]
    priceBook = get_slots(intent_request)["priceBook"]

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
        
    intent_request["sessionAttributes"] = {"marketCap": marketCap, 
        "tradeVolume": tradeVolume, "dividendPayer": dividendPayer,
        "sharePrice": sharePrice, "priceEarning": priceEarning, 
        "priceSale": priceSale, "priceBook": priceBook}

    # Return a message with conversion's result.
    return close(
        intent_request["sessionAttributes"],
        "Fulfilled",
        {
            "contentType": "PlainText",
            "content": "Thank you for your information."
        },
    )

def selectorIntent(intent_request):
    """
    Performs dialog management and fulfillment for recommending a portfolio.
    """

    # Gets slots' values
    bollingerBand = get_slots(intent_request)["bollingerBand"]
    eMA = get_slots(intent_request)["eMA"]

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
    
    intent_request["sessionAttributes"] = {"bollingerBand": bollingerBand, 
        "eMA": eMA}

    # Return a message with conversion's result.
    return close(
        intent_request["sessionAttributes"],
        "Fulfilled",
        {
            "contentType": "PlainText",
            "content": """Thank you for your information;
            {}.
            """.format(
                "good selector info"
            ),
        },
    )

def defineIntent(intent_request):
    """
    Performs dialog management and fulfillment for recommending a portfolio.
    """

    # Gets slots' values
    timeFrame = get_slots(intent_request)["timeFrame"]
    portfolioSize = get_slots(intent_request)["portfolioSize"]

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
    
    intent_request["sessionAttributes"] = {"portfolioSize": portfolioSize, 
        "timeFrame": timeFrame}

    # Return a message with conversion's result.
    return close(
        intent_request["sessionAttributes"],
        "Fulfilled",
        {
            "contentType": "PlainText",
            "content": """Thank you for your information;
            {}.
            """.format(
                "good define info"
            ),
        },
    )

def MLIntent(intent_request):
    """
    Performs dialog management and fulfillment for recommending a portfolio.
    """
    tickers = get_symbols(api)
    prices_df = get_prices(tickers) # same as data csv
    
    
    # Gets slots' values
    mlPredictions = get_slots(intent_request)["mlPredictions"]

    # Gets the invocation source, for Lex dialogs "DialogCodeHook" is expected.
    source = intent_request["invocationSource"]  

    if source == "DialogCodeHook":
        # This code performs basic validation on the supplied input slots.

        # Gets all the slots
        slots = get_slots(intent_request)
        
        print(slots)
        
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

        # take valid values from slots, convert to variables, use variables to filter dataframe
        marketCapReturn = intent_request["currentIntent"]["slots"]["marketcap"]
        mCR_df = prices_df[prices_df["marketcap"] == marketCapReturn]
        
        tradeVolumeReturn = intent_request["currentIntent"]["slots"]["volume"]
        tVR_df = mCR_df[prices_df["volume"] == tradeVolumeReturn]
        
        sharePriceReturn = intent_request["currentIntent"]["slots"]["close"]
        sPR_df = pBR_df[prices_df["close"] == sharePriceReturn]
        
        priceBookReturn = intent_request["currentIntent"]["slots"]["pb"]
        pBR_df = tVR_df[prices_df["pb"] == priceBookReturn]
        
        priceSaleReturn = intent_request["currentIntent"]["slots"]["ps"]
        pSR_df = pBR_df[prices_df["ps"] == priceSaleReturn]
        
        priceEarningReturn = intent_request["currentIntent"]["slots"]["pe"]
        pER_df = pSR_df[prices_df["pe"] == priceEarningReturn]

        

        # Once all slots are valid, a delegate dialog is returned to Lex to choose the next course of action.
        return delegate(output_session_attributes, get_slots(intent_request))
    
    # intent_request["sessionAttributes"] = {"mlPredictions": mlPredictions}

    # Return a message with conversion's result.
    return close(
        intent_request["sessionAttributes"],
        "Fulfilled",
        {
            "contentType": "PlainText",
            "content": "Return message"
        }
    ), pER_df

### Intents Dispatcher ###
def dispatch(intent_request):
    """
    Called when the user specifies an intent for this bot.
    """

    # Get the name of the current intent
    intent_name = intent_request["currentIntent"]["name"]

    # Dispatch to bot's intent handlers
    if intent_name == "userProfile":
        return userProfile(intent_request)
    elif intent_name == "filterIntent":
        return filterIntent(intent_request)
    elif intent_name == "selectorIntent":
        return selectorIntent(intent_request)
    elif intent_name == "defineIntent":
        return defineIntent(intent_request)
    elif intent_name == "MLIntent":
        return MLIntent(intent_request)

    raise Exception("Intent with name " + intent_name + " not supported")


### Main Handler ###
def lambda_handler(event, context):
    """
    Route the incoming request based on intent.
    The JSON body of the request is provided in the event slot.
    """

    return dispatch(event)
