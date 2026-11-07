selected_categories = [29, 16, 4]
CATEGORY_CONFIG = {
    29: {"category_name": "bread", "shelf_life": 1, "price": 2.00, "cost": 0.80},
    16: {"category_name": "meat",    "shelf_life": 3, "price": 6.00, "cost": 4.00},
    4:  {"category_name": "vegetables",   "shelf_life": 5, "price": 2.50, "cost": 1.50},
}
SALVAGE = 0.0
FIFO_SHARE = 0.7      # 70% of customers take the oldest stock
HOLDING_COST = 0.0
CAL_DAYS = 14

TAU = {29: 0.60, 16: 0.75, 4: 0.88}   # target quantile per category; move to retail_configs

CAT_COLS = ["store_id", "product_id", "first_category_id", "third_category_id"]
NUM_COLS = ["lag_1", "lag_7", "mean_7", "mean_28", "std_7", "weekday",
            "discount", "holiday_flag", "activity_flag",
            "precpt", "avg_temperature", "avg_humidity", "avg_wind_level"]
FEATURES = NUM_COLS + CAT_COLS
TARGET = "sale_amount_recovered"
