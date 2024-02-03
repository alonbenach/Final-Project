# %%
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import statsmodels.tools.tools as ct
import statsmodels.api as sm

# %%
### Import all relevant data
# define a range of dates to collect the data
start_date = "2000-12-01"
end_date = "2022-12-31"

# create a list of stock tickers
tickers = [
    "AMD",
    "KO",
    "LMT",
    "SBUX",
    "GE",
    "AMGN",
    "CAT",
    "HD",
    "JNJ",
    "BA",
    "TGT",
    "XOM",
    "PG",
    "DIS",
    "ABT",
    "MS",
    "F",
    "MCD",
    "MO",
    "MMM",
    "WBA",
    "AXP",
    "T",
    "RTX",
    "PFE",
    "VZ",
    "BMY",
    "KR",
    "CL",
    "DUK",
    "HON",
    "ZBRA",
    "UPS",
    "TXN",
    "SYY",
    "DD",
    "CLX",
    "MCK",
    "JCI",
    "HSY",
]

# source the data for all stocks
stock_prices = pd.DataFrame()
for ticker in tickers:
    stock = yf.download(ticker, start=start_date, end=end_date, interval="1mo")["Close"]
    stock_prices[ticker] = stock

# store as Excel file
file_name = "stock_prices_Dec2000_Dec2022.xlsx"

stock_prices.to_excel(file_name, index=True)


### Transform the data to be used for analysis and save it in a separate file
# Check that the Date column serves as index
print(stock_prices.index)

# Create a df of monthly returns
monthly_returns = stock_prices.pct_change().dropna()
print(f"\nThe monthly return for each stock in my portfolio: \n {monthly_returns}")

# Reindexing to end-of-month dates
monthly_returns.index = pd.to_datetime(monthly_returns.index)
monthly_returns = monthly_returns.resample("M").last()


# Save both variables in xlsx format
with pd.ExcelWriter("Benach_IT_final_data.xlsx") as writer:
    monthly_returns.to_excel(writer, sheet_name="monthly_returns", index=True)

# %%
############## Load all dataframes for analysis (if data already created, begin run from here) ##############
f = "Benach_IT_final_data.xlsx"
ret_df = pd.read_excel(f, parse_dates=["Date"], index_col="Date")


f = "F-F_Momentum_Factor.csv"
momentum = pd.read_csv(f, parse_dates=["Date"], index_col="Date")

f = "F-F_Research_Data_5_Factors_2x3.csv"
ff_df = pd.read_csv(f, parse_dates=["Date"], index_col="Date")

ff_df["MOM"] = momentum.loc[:, "Mom"]
ff_df = ff_df / 100
ff_df.index = ff_df.index.strftime("%Y-%m-%d")
ff_df.index = pd.to_datetime(ff_df.index)


# %%
############################# Short Run Selection Period #############################
# Construct and analyze the risk adjusted performance of the 40 individual stocks

# Merge the stock returns and Fama-French factors data
merged_df = pd.merge(ret_df, ff_df, left_index=True, right_index=True)
market_betas_df = pd.DataFrame(index=ret_df.columns, columns=ret_df.index)

# Step 1: Calculate excess returns for each stock
for stock in ret_df.columns:
    merged_df[f"{stock}_Excess_Return"] = ret_df[stock] - merged_df["RF"]

# Step 2: Define the independent variables (factors)
independent_variables = ["Mkt-RF", "SMB", "HML", "MOM"]

# Step 3: Assess each stock over a 12-month rolling window
rolling_window = 12
for i in range(rolling_window, len(ret_df) + 1):
    window_data = merged_df.iloc[i - rolling_window : i]

    for stock in ret_df.columns:
        y = window_data[f"{stock}_Excess_Return"]
        X = sm.add_constant(window_data[independent_variables])

        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})

        # Store market beta in market_betas_df
        market_beta = model.params["Mkt-RF"]
        market_betas_df.loc[stock, window_data.index[-1]] = market_beta

market_betas_df = market_betas_df.dropna(axis=1)

print(f"There are {market_betas_df.isna().mean().sum()} NA values in market_betas_df")

# Convert the data to numeric to handle non-numeric values
market_betas_df = market_betas_df.apply(pd.to_numeric, errors="coerce")

# %%
# Step 4: Split stocks into quintiles for each period
quintile_dictionaries = {}
for date in market_betas_df.columns:
    quintile_dict = {}

    # Create quintiles
    bins = pd.qcut(
        market_betas_df[date],
        q=[0, 0.2, 0.4, 0.6, 0.8, 1],
        labels=False,
        duplicates="drop",
    )

    # Store stocks in each quintile
    for quintile in range(0, 5):
        quintile_dict[quintile + 1] = market_betas_df.index[bins == quintile].tolist()

    quintile_dictionaries[date.date().strftime("%Y-%m-%d")] = quintile_dict

# %%
# Step 5a: Calculate quintile portfolios as the simple average - short term (yearly rebalanced)

# Initialize the portfolio_returns DataFrame with the calculated index
portfolios_dictionary_yearly = pd.DataFrame(columns=range(1, 6))

fixing_window = 12

# Group keys into chunks of 12
for i in range(0, len(quintile_dictionaries), fixing_window):
    twelve_keys = list(quintile_dictionaries.keys())[i : i + fixing_window]
    first_key_content = quintile_dictionaries[twelve_keys[0]]
    print(i, twelve_keys)

    # Assign the repeated content to each key in the group
    for key in twelve_keys:
        portfolios_dictionary_yearly[key] = first_key_content

portfolios_dictionary_yearly = portfolios_dictionary_yearly.dropna(axis=1)

portfolios_df_yearly = pd.DataFrame(columns=range(1, 6))

for date, tickers in portfolios_dictionary_yearly.items():
    # Get the following month's end date
    end_of_next_month = pd.to_datetime(date) + pd.tseries.offsets.MonthEnd(1)
    print(end_of_next_month)

    try:
        for i in range(1, len(tickers) + 1):
            # Get the returns for the end of the following month for the specified tickers
            next_month_returns = ret_df.loc[str(end_of_next_month.date()), tickers[i]]

            # Calculate the average return for the next month
            average_return = next_month_returns.mean()
            # Store the average return in the DataFrame
            portfolios_df_yearly.loc[str(end_of_next_month.date()), i] = average_return
            print(portfolios_df_yearly)

    except KeyError as e:
        print(f"No data found for {end_of_next_month.date()}: {e}")

portfolios_df_yearly.index = pd.to_datetime(portfolios_df_yearly.index)

# %%
# Step 5b: Calculate quintile portfolios as the simple average - short term (yearly rebalanced)

# Initialize the portfolio_returns DataFrame with the calculated index
portfolios_dictionary_3_years = pd.DataFrame(columns=range(1, 6))

fixing_window = 36

# Group keys into chunks of 12
for i in range(0, len(quintile_dictionaries), fixing_window):
    keys_36 = list(quintile_dictionaries.keys())[i : i + fixing_window]
    first_key_content = quintile_dictionaries[keys_36[0]]
    print(i, keys_36)

    # Assign the repeated content to each member of the group
    for key in keys_36:
        portfolios_dictionary_3_years[key] = first_key_content

portfolios_dictionary_3_years = portfolios_dictionary_3_years.dropna(axis=1)

portfolios_df_3_years = pd.DataFrame(columns=range(1, 6))

for date, tickers in portfolios_dictionary_3_years.items():
    # Get the following month's end date
    end_of_next_month = pd.to_datetime(date) + pd.tseries.offsets.MonthEnd(1)
    print(end_of_next_month)

    try:
        for i in range(1, len(tickers) + 1):
            # Get the returns for the end of the following month for the specified tickers
            next_month_returns = ret_df.loc[str(end_of_next_month.date()), tickers[i]]

            # Calculate the average return for the next month
            average_return = next_month_returns.mean()
            # Store the average return in the DataFrame
            portfolios_df_3_years.loc[str(end_of_next_month.date()), i] = average_return
            print(portfolios_df_3_years)

    except KeyError as e:
        print(f"No data found for {end_of_next_month.date()}: {e}")

portfolios_df_3_years.index = pd.to_datetime(portfolios_df_3_years.index)

############################### Performance Asessment ###############################
# %%
# Short term performance with portfolios_df_yearly
# Calculate cumulative returns
cumulative_returns_yearly_df = (1 + portfolios_df_yearly).cumprod() - 1

# Plot cumulative returns
plt.figure(figsize=(16, 8))
cumulative_returns_yearly_df.plot(ax=plt.gca(), grid=True, colormap="Set1")

# Format y-axis as percentage
plt.gca().set_yticks(plt.gca().get_yticks())
plt.gca().set_yticklabels(["{:.0%}".format(x) for x in plt.gca().get_yticks()])

# Add grid
plt.grid(True, linestyle="--", alpha=0.7)

plt.title("Cumulative Returns of Portfolios - Short Investment")
plt.xlabel("Date")
plt.ylabel("Cumulative Return of 1 Year Rebalancing Intervals")
plt.legend(title="Portfolios Beta Quintiles")

# Set a nice and futuristic background color
plt.gca().set_facecolor("#f5f5f5")

plt.show()

# %%
# Calculate performance metrics for each portfolio
# Declare an empty DataFrame with the desired column order
columns_order = [
    "Average Return",
    "Standard Deviation",
    "Average Excess Return",
    "Sharpe Ratio",
    "Alpha",
    "PV_Alpha",
    "Beta_Mkt-RF",
    "PV_Mkt-RF",
    "Beta_SMB",
    "PV_SMB",
    "Beta_HML",
    "PV_HML",
    "Beta_MOM",
    "PV_MOM",
    "R-squared",
]

performance_metrics_df_yearly = pd.DataFrame(columns=columns_order)

# Calculate annualized average return
average_return = portfolios_df_yearly.mean()

# Calculate annualized standard deviation
std_deviation = portfolios_df_yearly.std()

# Calculate average excess return
portfolios_df_yearly_excess_ret = portfolios_df_yearly.sub(
    merged_df["RF"][portfolios_df_yearly.index], axis=0
)
average_excess_return = portfolios_df_yearly_excess_ret.mean()


# Calculate Sharpe ratio
sharpe_ratio = average_excess_return / std_deviation

# Create a DataFrame for performance metrics
performance_metrics_df_yearly = pd.DataFrame(
    {
        "Average Return": average_return,
        "Standard Deviation": std_deviation,
        "Average Excess Return": average_excess_return,
        "Sharpe Ratio": sharpe_ratio,
    }
)

# Print the performance metrics DataFrame
print("Performance Metrics:")
print(performance_metrics_df_yearly)

# %%
# Add Carhart model metrics
# Define independent variables
independent_variables = ["Mkt-RF", "SMB", "HML", "MOM"]

# Loop through each portfolio
for portfolio in portfolios_df_yearly_excess_ret.columns:
    # Get the dependent variable (excess return of the portfolio)
    y = portfolios_df_yearly_excess_ret[portfolio].astype(float)

    # Get the independent variables from merged_df with the same index
    X = merged_df.loc[y.index, independent_variables]

    # Add a constant term to the independent variables
    X = sm.add_constant(X).astype(float)

    # Fit the OLS regression model
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})

    # Add regression results to performance_metrics_df_yearly
    performance_metrics_df_yearly.loc[portfolio, "Alpha"] = model.params["const"]
    performance_metrics_df_yearly.loc[portfolio, "PV_Alpha"] = model.pvalues["const"]
    performance_metrics_df_yearly.loc[portfolio, "Beta_Mkt-RF"] = model.params["Mkt-RF"]
    performance_metrics_df_yearly.loc[portfolio, "PV_Mkt-RF"] = model.pvalues["Mkt-RF"]
    performance_metrics_df_yearly.loc[portfolio, "Beta_SMB"] = model.params["SMB"]
    performance_metrics_df_yearly.loc[portfolio, "PV_SMB"] = model.pvalues["SMB"]
    performance_metrics_df_yearly.loc[portfolio, "Beta_HML"] = model.params["HML"]
    performance_metrics_df_yearly.loc[portfolio, "PV_HML"] = model.pvalues["HML"]
    performance_metrics_df_yearly.loc[portfolio, "Beta_MOM"] = model.params["MOM"]
    performance_metrics_df_yearly.loc[portfolio, "PV_MOM"] = model.pvalues["MOM"]
    performance_metrics_df_yearly.loc[portfolio, "R-squared"] = model.rsquared

# Print the OLS regression results
print("OLS Regression Results:")
print(performance_metrics_df_yearly)

# %%
# Define the Excel file name
excel_file_name = "1_year_selection_1_year_performance.xlsx"

# Create ExcelWriter
with pd.ExcelWriter(excel_file_name, engine="xlsxwriter") as writer:
    # Save performance_metrics_df_yearly to the 'performance metrics' sheet
    performance_metrics_df_yearly.to_excel(writer, sheet_name="performance metrics")

    # Save portfolios_df_yearly to the 'returns' sheet
    portfolios_df_yearly.to_excel(writer, sheet_name="returns")

    # Save portfolios_df_yearly_excess_ret to the 'excess returns' sheet
    portfolios_df_yearly_excess_ret.to_excel(writer, sheet_name="excess returns")

    # Save cumulative_returns_yearly_df to the 'cumret' sheet
    cumulative_returns_yearly_df.to_excel(writer, sheet_name="cum_ret")

# %%
# Short term performance with portfolios_df_3_years
# Calculate cumulative returns
cumulative_returns_3_year_df = (1 + portfolios_df_3_years).cumprod() - 1

# Plot cumulative returns
plt.figure(figsize=(16, 8))
cumulative_returns_3_year_df.plot(ax=plt.gca(), grid=True, colormap="Set1")

# Format y-axis as percentage
plt.gca().set_yticks(plt.gca().get_yticks())
plt.gca().set_yticklabels(["{:.0%}".format(x) for x in plt.gca().get_yticks()])

# Add grid
plt.grid(True, linestyle="--", alpha=0.7)

plt.title("Cumulative Returns of Portfolios - Long Investment")
plt.xlabel("Date")
plt.ylabel("Cumulative Return of 3 Year Rebalancing Intervals")
plt.legend(title="Portfolios Beta Quintiles")

# Set a nice and futuristic background color
plt.gca().set_facecolor("#f5f5f5")

plt.show()

# %%
# Calculate performance metrics for each portfolio
# Declare an empty DataFrame with the desired column order
columns_order = [
    "Average Return",
    "Standard Deviation",
    "Average Excess Return",
    "Sharpe Ratio",
    "Alpha",
    "PV_Alpha",
    "Beta_Mkt-RF",
    "PV_Mkt-RF",
    "Beta_SMB",
    "PV_SMB",
    "Beta_HML",
    "PV_HML",
    "Beta_MOM",
    "PV_MOM",
    "R-squared",
]

performance_metrics_df_3_year = pd.DataFrame(columns=columns_order)

# Calculate annualized average return
average_return = portfolios_df_3_years.mean()

# Calculate annualized standard deviation
std_deviation = portfolios_df_3_years.std()

# Calculate average excess return
portfolios_df_3_year_excess_ret = portfolios_df_3_years.sub(
    merged_df["RF"][portfolios_df_3_years.index], axis=0
)
average_excess_return = portfolios_df_3_year_excess_ret.mean()


# Calculate Sharpe ratio
sharpe_ratio = average_excess_return / std_deviation

# Create a DataFrame for performance metrics
performance_metrics_df_3_year = pd.DataFrame(
    {
        "Average Return": average_return,
        "Standard Deviation": std_deviation,
        "Average Excess Return": average_excess_return,
        "Sharpe Ratio": sharpe_ratio,
    }
)

# Print the performance metrics DataFrame
print("Performance Metrics:")
print(performance_metrics_df_3_year)

# %%
# Add Carhart model metrics
# Define independent variables
independent_variables = ["Mkt-RF", "SMB", "HML", "MOM"]

# Loop through each portfolio
for portfolio in portfolios_df_3_year_excess_ret.columns:
    # Get the dependent variable (excess return of the portfolio)
    y = portfolios_df_3_year_excess_ret[portfolio].astype(float)

    # Get the independent variables from merged_df with the same index
    X = merged_df.loc[y.index, independent_variables]

    # Add a constant term to the independent variables
    X = sm.add_constant(X).astype(float)

    # Fit the OLS regression model
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})

    # Add regression results to performance_metrics_df_3_year
    performance_metrics_df_3_year.loc[portfolio, "Alpha"] = model.params["const"]
    performance_metrics_df_3_year.loc[portfolio, "PV_Alpha"] = model.pvalues["const"]
    performance_metrics_df_3_year.loc[portfolio, "Beta_Mkt-RF"] = model.params["Mkt-RF"]
    performance_metrics_df_3_year.loc[portfolio, "PV_Mkt-RF"] = model.pvalues["Mkt-RF"]
    performance_metrics_df_3_year.loc[portfolio, "Beta_SMB"] = model.params["SMB"]
    performance_metrics_df_3_year.loc[portfolio, "PV_SMB"] = model.pvalues["SMB"]
    performance_metrics_df_3_year.loc[portfolio, "Beta_HML"] = model.params["HML"]
    performance_metrics_df_3_year.loc[portfolio, "PV_HML"] = model.pvalues["HML"]
    performance_metrics_df_3_year.loc[portfolio, "Beta_MOM"] = model.params["MOM"]
    performance_metrics_df_3_year.loc[portfolio, "PV_MOM"] = model.pvalues["MOM"]
    performance_metrics_df_3_year.loc[portfolio, "R-squared"] = model.rsquared

# Print the OLS regression results
print("OLS Regression Results:")
print(performance_metrics_df_3_year)

# %%
# Define the Excel file name
excel_file_name = "1_year_selection_3_year_performance.xlsx"

# Create ExcelWriter
with pd.ExcelWriter(excel_file_name, engine="xlsxwriter") as writer:
    # Save performance_metrics_df_3_year to the 'performance metrics' sheet
    performance_metrics_df_3_year.to_excel(writer, sheet_name="performance metrics")

    # Save portfolios_df_3_years to the 'returns' sheet
    portfolios_df_3_years.to_excel(writer, sheet_name="returns")

    # Save portfolios_df_yearly_excess_ret to the 'excess returns' sheet
    portfolios_df_3_year_excess_ret.to_excel(writer, sheet_name="excess returns")

    # Save cumulative_returns_3_year_df to the 'cumret' sheet
    cumulative_returns_3_year_df.to_excel(writer, sheet_name="cum_ret")

# %%
