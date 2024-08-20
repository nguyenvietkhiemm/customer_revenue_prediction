from pyspark.sql import functions as F
from modules.aggregation import *
### CREATE MONTH FEATURE

def add_months_to_end(df, months=12):
    max_date = df.agg(F.max("date")).collect()[0][0]
    date_diff = F.datediff(F.lit(max_date), F.col("date"))
    months_to_end = (date_diff / 30).cast(IntegerType()) + 1
    months_to_end = F.when(months_to_end > months, None).otherwise(months_to_end)
    return df.withColumn("months_to_end", months_to_end)

### CREATE DATE FEATURES

def encode_date(df):
    attrs = ['Dayofweek']
    for attr in attrs:
        df = df.withColumn('date_' + attr, F.expr("EXTRACT(DOW FROM date)").cast(IntegerType()))
    return df

### WRAPPER FEATURE ENGINEERING FUNCTION

def create_data(df, x_idx, y_idx, old_months=7, agg_months=1):
    ###### PHÂN CHIA DỮ LIỆU
    _df = df.withColumn("index", F.monotonically_increasing_id())
    df_selected = _df.filter(F.col("index").isin(x_idx))
    ordered_date = df_selected.select(F.col("date").alias("ordered_date")).orderBy("ordered_date").distinct()
    
    old_x_min_date = ordered_date.first()["ordered_date"] - pd.DateOffset(months=old_months)
    x = _df.filter((F.col("date") >= old_x_min_date) & (F.col("index").isin(x_idx))).drop("index")

    ###### ENGINEERING CÁC ĐẶC TRƯNG
     
    ### Tính toán các đặc trưng tổng hợp 
    grp_x = aggregate_data(x, group_var=["fullVisitorId"])
    
    # Số lần ghé thăm
    visit_number_cols = [col for col in grp_x.columns if "visitNumber" in col]
    grp_x = grp_x.drop(*visit_number_cols)
    
    # lấy max vì nó là log ghi lại số lần ghé thăm cho mỗi session
    num_visits_df = x.groupBy("fullVisitorId").agg(F.max("visitNumber").alias("num_visits"))
    grp_x = grp_x.join(num_visits_df, on="fullVisitorId", how="left")
    
    # Tính "recency" (thời gian từ lần ghé thăm gần nhất)
    max_date = x.agg(F.max("date")).collect()[0][0]
    recency_df = x.groupBy("fullVisitorId").agg(
        F.datediff(F.lit(max_date), F.max("date")).alias("recency"))
    recency_df = recency_df.withColumn("recency", F.col("recency").cast(IntegerType()))
    grp_x = grp_x.join(recency_df, on="fullVisitorId", how="left")

    # Tính "frequency" (tần suất ghé thăm)
    frequency_df = x.groupBy("fullVisitorId").agg(F.count("date").alias("frequency"))
    grp_x = grp_x.join(frequency_df, on="fullVisitorId", how="left")
    grp_x = grp_x.withColumn("frequency", F.coalesce(F.col("frequency"), F.lit(0)))

    ###### CĂN CHỈNH DỮ LIỆU
    
    if len(y_idx) > 0:
        y = df.filter(F.col("fullVisitorId").isin(y_idx)).groupBy("fullVisitorId").agg(F.log1p(F.sum("totals_transactionRevenue")).alias("target"))
        new_ids_df = grp_x.join(y, on="fullVisitorId", how="left_anti").select("fullVisitorId")

        y = y.union(new_ids_df.withColumn("target", F.lit(0)))

        y = y.orderBy("fullVisitorId").select("target")
        x = grp_x.orderBy("fullVisitorId")
        return x, y

    else:
        x = grp_x.orderBy("fullVisitorId")
        return x

