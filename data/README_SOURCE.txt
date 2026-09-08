Demand calibration source

daily_demand_real.csv and demand_calibration.json are derived from:

  ../../Online Retail Project/data/cleaned_data_with_customer.csv

StockCode 85048, "15CM CHRISTMAS GLASS BALL 20 LIGHTS", filtered to
positive quantity, non cancellation rows, resampled to daily totals.
Original data: Online Retail II, UCI Machine Learning Repository.

Regenerate with: python3 src/demand_calibration.py
(reads the path above directly, no copy is kept here except the two
derived files listed above, so the Online Retail Project folder needs to
stay in place for this to be re-run from scratch)
