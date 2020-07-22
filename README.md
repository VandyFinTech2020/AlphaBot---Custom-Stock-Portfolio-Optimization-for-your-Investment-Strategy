# AlphaBot - Custom Stock Portfolio Optimization for your Investment Strategy

## Background

Harold has recently been promoted to portfolio manager for a state pension fund. To support the growing number of retired public employees, he needs to generate strong investment returns. However, the COVID-19 pandemic has forced the state to reduce headcount across departments, which includes Harold's investment teams. In order to meet his mandate with reduced resources, Harold realizes he will need to rely much more on technology portfolio selection. US-listed (NYSE or NASDAQ) stocks. 

He has tasked our team to develop a bot that will allow him to input his selection criteria and then compute an optimally-weighted portfolio of US-listed (NYSE and NASDAQ). The state has agreed to fund our work as long as it is user-friendly enough that it could potentially be offered to state employees for their own personal portfolio construction. 

## AlphaBot Intent and Slots

Our team has determined an AWS bot programmed with Amazon Lex and Lambda is best suited to devise a portfolio from his selection criteria. We have created an intent to process user selection criteria and produce an initial portfolio:

### Intent :  'MLIntent'
This intent allows the user to trim the universe of US-listed stocks based on fund restrictions.

#### Slot_1:  First Name
* for personalized bot interactions

#### Slot_2: User Age
* to verify the user is of majority age (21)
    
#### Slot_3: Contact Info
* user must input 10-digit phone number

#### Slot_4:  Market Cap
* Card Buttons
    1. Small Cap and Under (< 2 billion)
    2. Mid Cap (> 2 billion and < 10 billion )
    3. Large Cap (> 10 billion and < 200 billion )
    4. Mega Cap (> 200 billion)

#### Slot_5:  Thinly-Traded (ADTV < 100,000 shares) Allowed? 
* Card Buttons
    1. No
    2. Yes
    
#### Slot_6:  Share Price Floor
* Card Buttons
    1. > 5.00 per share
    2. > 1.00 per share
    3. None
    
#### Slot_7:  Price / Earnings (TTM)
* Card Buttons
    1. NM
    2. Low (> 0 and < 10 )
    3. Average ( > 10 and < 20)
    4. High (> 20)
    

#### Slot_8:  Price / Sales (TTM)
* Card Buttons
    1. Low ( < 1 )
    2. Fair (> 1 and < 2 )
    3. High (> 2)
  

#### Slot_9: Price / Book
* Card Buttons
    1. Low (< 1)
    2. Fair (> 1 and < 2)
    3. High (> 2)
    
#### Slot_10:  Timeframe
    -User can decide to select the default selection of 2000 days or choose a different horizon.

#### Confirmation Prompt
* If some number of stocks meets all of the above criteria, then the user is told:
    "Cool, we have stocks"
* Otherwise the user is informed:
    "No stocks met your criteria."
 
### Output
* Our lambda code includes unchained Bollinger Band and Exponential Moving Average code so a user can apply further manual filtering, if desired.
* The user takes the list of stocks from MLIntent and runs them through our LSTM model.

## Determining Expected Return - LSTM Model
* This model will produce a list of 4-10 stocks that are expected to produce the greatest expected return over the selected timeframe.
