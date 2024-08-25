#!/usr/bin/env python
# coding: utf-8

# Import libs
import json
import os
from datetime import datetime, timedelta

import yfinance as yf
from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

# Creating Stock Analyzer AI Tool
def fetch_stock_price(ticket, dt_start, dt_end):
    stock = yf.download(ticket, dt_start, dt_end)
    return stock

yahoo_finance_tool = Tool(
    name = "Stock Analyzer AI",
    description = "Fetches stock prices for {ticket} from the last year about a specific company from Yahoo Finance API.",
    func = lambda ticket: fetch_stock_price(ticket, dt_start, dt_end)
)

# Importing LLM GPT of OpenAI
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(
    model="gpt-3.5-turbo"
    )

# Creating Stock Price Analyst Agent
stockPriceAnalystAgent = Agent(
    role = "Senior Stock Price Analyst",
    goal = "Find the {ticket} stock price and analyses trends",
    backstory = """
    You're a highly experienced in analyzing the price of a specific stock and make
    predictions about its future price.
    """,
    verbose = True,
    llm = llm,
    max_iter = 3,
    memory = True,
    allow_delegation = False,
    tools = [yahoo_finance_tool]
)

# Creating Stock Price Analysis Task
getStockPrice = Task(
    description = "Analyze de stock {ticket} price history and create a trend analyses of up, down or sideways",
    agent = stockPriceAnalystAgent,
    expected_output = """
    Specify the current trend stock price - up, down or sideways.
     eg. stock = 'AAPL', price UP.
     """,
)

# Creating News Stock Searching Tool
search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)

# Creating Stock News Analyst Agent
newsStockAnalystAgent = Agent(
    role = "Stock News Analyst",
    goal = """
    Create a short summary of the market news related to the stock {ticket} company.
    Specify the current trend - up, down or sideways with the news context. For each requested stock 
    asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.

    Always remember to include news headlines and news links in your summary.
    """,
    backstory = """
    You're a highly experienced in analyzing the market trends and news and have 
    tracked assets for more then 20 years.

    You're also master level analyst in the traditional markets and have deep understanding of human 
    psychology.
    
    You understand news, theirs tittles and theirs information, but you look at those with a health 
    dose of skepticism. You consider also the source of the news articles.
    """,
    verbose = True,
    llm = llm,
    max_iter = 7,
    memory = True,
    allow_delegation = False,
    tools = [search_tool]
)

# Creating Stock News Analysis Task
getStockNews = Task(
    description = f"""
    Take the stock and always include BTC to it (if no request).
    Use the search tool to search each one individually.
    
    The current date is {datetime.now().strftime('%d/%m/%Y')}.
    
    Compose the result into a helpful report.
    """,
    agent = newsStockAnalystAgent,
    expected_output = """
    A summary of the overall market and one sentence summary for each request 
    asset.
    
    Include a fear/greed score for each asset based on the news. Use format:
    - <STOCK ASSET>
    - <SUMMARY BASED OF NEWS>
    - <TREND PREDICTION>
    - <FEAR/GREED SCORE>
    - <NEWS HEADLINE>
    - <URL OF EVERY NEWS>
    """,
)

# Creating Stock Analyst Writer Agent
stockAnalystWriterAgent = Agent(
    role = "Senior Stock Analyst Writer",
    goal = """
    Analyse the trends price and news and write an insightful compelling and informative 3 paragraph long newsletter based on the 
    stock report and price trend.
    """,
    backstory = """
    You're widely accepted as the best stock analyst in the market. You understand 
    complex concepts and create compelling stories and narratives that resonate withe the audiences.
    
    You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses. 
    You're able to hold multiple opinions when analyzing anything data.
    """,
    verbose = True,
    llm = llm,
    max_iter = 3,
    memory = True,
    allow_delegation = True
)

# Creating Stock Analyst Writer Task
writeAnalyses = Task(
    description = """
    Use the stock price trend and the stock news report to create an analyses and write the newsletter 
    about the {ticket} company that is brief and highlights the most important points.

    Focus on the stock price trend, news and fear/greed score. What are the near future considerations?

    Include the previous analyses of stock trend and news summary.
    """,
    agent = stockAnalystWriterAgent,
    expected_output = """An eloquent 3 paragraphs newsletter format as markdown in a easy readable manner. 
    Its should contain:

    - 3 bullets executive summary
    - Introduction - set the overall picture and spike up the interest 
    - main part provides the meat of the analyses including the news summary and fear/greed score.
    - summary - key facts and concrete future trends prediction - up, down, sideways
    
    """,
    context = [getStockPrice, getStockNews]
)

# Creating Crew
crew = Crew(
    agents = [stockPriceAnalystAgent, newsStockAnalystAgent, stockAnalystWriterAgent],
    tasks = [getStockPrice, getStockNews, writeAnalyses],
    verbose = 2,
    process = Process.hierarchical,
    full_output = True,
    share_crew = False,
    manager_llm = llm,
    max_iter = 10
)
#########################################################################
#
# Beginning of web configuration with streamlit
#
#########################################################################
# Defining variables for Stock Analyzer
current_date = datetime.now().date()
max_date_research = current_date - timedelta(days=10)
min_date_research = current_date - timedelta(days=40)
limit_date_research = current_date - timedelta(days=365)

# Configuration of the web app
st.set_page_config(
    page_title="Stock Analyzer AI",
    page_icon="./img/stock-analyzer-ai.png"
)
col1, col2 = st.columns([1, 3])
with col1:
    st.image("./img/stock-analyzer-ai.png", use_column_width=True)
with col2:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.title("Stock Analyzer AI")

st.subheader("Description App", divider = True)
st.write("""
    <div style="text-align: justify;">
        Stock Analyzer AI is a powerful web app that simplifies stock market analysis <b>using advanced AI</b>. It integrates <i>OpenAI, Yahoo Finance, and DuckDuckGoSearchResults</i> to deliver a comprehensive view of any stock. The app <b>retrieves historical stock data and finds relevant news</b>, allowing you to see trends and understand market context. Then, it performs in-depth analysis using AI to provide insights into how these factors may influence future stock behavior.
    </div>
         """, unsafe_allow_html=True
)
st.write("")
st.write("""
    <div style="text-align: justify;">         
Whether you are a novice investor or a seasoned pro, Stock Analyzer AI takes complex financial data into an easy-to-understand format, saving you time searching for information. Explore Stock Analyzer AI today and get an edge in the stock market!
</div>
         """, unsafe_allow_html=True
)
st.write("")
st.write("""
    <div style="text-align: justify;">     
<b>Please note:</b> This app facilitates the analysis of a stock based on the period defined in the search and the most recent news. Under no circumstances should you take the results as an investment or disinvestment recommendation.!
    </div>
    """, unsafe_allow_html=True
)

st.logo("./img/stock-analyzer-ai.png")

with st.sidebar:
    st.header("Stock and period to Research")

    with st.form(key="research_form"):
        ticket = st.text_input("Enter the ticket (stock code eg. AAPL, TSLA, TSM, AMZN, GOOGL, etc.)")
        dt_start = st.date_input(
            "Start date for analyse", 
            value = min_date_research,
            max_value = max_date_research,
            min_value = limit_date_research,
            format = "YYYY-MM-DD", 
            disabled = False, 
            label_visibility = "visible"
            )
        dt_end = st.date_input(
            "End date for analyse", 
            value = max_date_research,
            max_value = max_date_research,
            min_value = limit_date_research,
            format = "YYYY-MM-DD", 
            disabled = False, 
            label_visibility = "visible"
            )
        submit_button = st.form_submit_button(label="Run Research")

if submit_button:
    if not ticket:
        st.write("")
        st.error("Please fill the ticket field")
    elif dt_start > dt_end:
        st.write("")
        st.error("Start date must be lower than End date")
    else:
        results = crew.kickoff(inputs={"ticket": ticket, "dt_start": dt_start, "dt_end": dt_end})
        st.write("")
        st.subheader("Result of your research", divider = True)
        st.write(results['final_output'])
        st.write("")
        st.write(
            """
            <div style="text-align: justify;">
                <i>The above result was generated thanks to the analysis of Senior Stock 
                 Price Analyst Agent and Stock News Analyst Agent :clap:</i>
            </div>	
            """, unsafe_allow_html=True
        )

        st.subheader("Result of the Senior Stock Price Analyst Agent :moneybag:", divider = True)
        st.write(results['tasks_outputs'][0].exported_output)

        st.subheader("Result of the Stock News Analyst Agent :newspaper:", divider = True)
        st.write(results['tasks_outputs'][1].exported_output)
        
        st.write("")
        st.write("""
            <div style="text-align: justify;">
            <b>Please remember: Under no circumstances should you take the results as an investment or disinvestment recommendation.!</b>
            </div>
        """, unsafe_allow_html=True
        )