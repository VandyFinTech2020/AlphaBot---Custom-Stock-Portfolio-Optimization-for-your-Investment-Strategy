# AlphaBot - Custom Stock Portfolio Optimization for your Investment Strategy

## Background

Harold has recently been promoted to portfolio manager for a state pension fund. To support the growing number of retired public employees, he needs to generate strong investment returns. However, the COVID-19 pandemic has forced the state to reduce headcount across departments, which includes Harold's investment teams. In order to meet his mandate with reduced resources, Harold realizes he will need to rely much more on technology to design and dynamically maintain a portfolio of US-listed (NYSE or NASDAQ) stocks. 

He has tasked our team to develop a process that will allow him to input his selection criteria and compute an optimally-weighted portfolio that will produce the best possible returns for his retirees. The state has agreed to fund our work as long as other state portfolio managers will be able to use it to meet their own investment mandates.

## AlphaBot Intents and Slots

Our team has determined an AWS bot programmed with Amazon Lex and Lambda is best suited to devise a portfolio from his selection criteria. We have created three intents to process his inputs and produce an initial portfolio:

### Intent_1 :  'filterStocks'
This intent allows the user to trim the universe of US-listed stocks based on fund restrictions. 

#### Slot_1:  Market Cap
* Card Buttons
    1. Small Cap (> 300 million and < 2 billion )
    2. Mid Cap (> 2 billion and < 10 billion )
    3. Large Cap (> 10 billion and < 200 billion )
    4. Mega Cap (> 200 billion)

#### Slot_2:  Thinly-Traded (ADTV < 100,000 shares) Allowed? 
* Card Buttons
    1. No
    2. Yes

#### Slot_3:  Dividend Payer
* Card Buttons
    1. No
    2. Yes
    
#### Slot_4:  Share Price Floor
* Card Buttons
    1. > 5.00 per share
    2. > 1.00 per share
    3. None

filterStocks will output a curated list of acceptable listed stocks based on user inputs.

### Intent_2 :  'stockSelect'
This intent allows the user to trim the universe of US-listed stocks based on fund restrictions. 

#### Slot_1:  Price / Sales (TTM)
* Card Buttons
    1. Low ( < 1 )
    2. Fair (> 1 and < 2 )
    3. High (> 2)
  
#### Slot_2:  Price / Earnings (TTM)
* Card Buttons
    1. NM
    2. Low (> 0 and < 10 )
    3. Below Average ( > 10 and < 13.5)
    4. Average (> 13.5 and < 16.5)
    5. Above Average (> 16.5 and < 20)
    6. High (> 20)
    
#### Slot_3: Price / Book
* Card Buttons
    1. Low (< 1)
    2. Fair (> 1 and < 2)
    3. High (> 2)
    
#### Slot_3:  Bollinger Band Positioning (2 std deviations +/- 20-day SMA)
* Card Buttons
    1. Overbought (Trading Above Upper Band)
    2. Trading in Range
    3. Oversold (Trading Below Lower Band)
    
#### Slot_4: Exponential Moving Average Positioning  
* Card Buttons
    1. Bullish - Golden Cross Zone (50 EMA has broken above 200 EMA)
    2. Bearish - Death Cross Zone (50 EMA has broken below 200 EMA)
    
stockSelect will output a ranked list of potential stocks

### Intent_3 :  'holdingRange'

#### Slot_1: Expected Timeframe
* Card Buttons
    1. 1 Week
    2. 1 Month
    3. 3 Months (1 quarter)
    4. 6 Months
    5. 1 Year
    6. 2 Years

holdingRange will be incorporated in ML model to rank potential portfolio stocks

?Vader sentiment analysis for 'goodness' (longs) and 'badness' (shorts)

*Machine Learning modeling portion on how to retrain model TBA*
