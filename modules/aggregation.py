from pyspark.sql import functions as F
from pyspark.sql import Window

def aggregate_data(df, 
                   group_var, 
                   num_stats = [ 'count', 'mean', 'sum'], 
                   label     = None, 
                   sd_zeros  = False): 
    
    # Find factor columns
    df_factors = [ 'channelGrouping', 'date', 'fullVisitorId', 'visitId', 'device_operatingSystem', 'geoNetwork_country',
                  'customDimensions_index', 'device_browser', 'device_isMobile', 'device_deviceCategory', 'geoNetwork_continent',
                  'geoNetwork_subContinent', 'geoNetwork_region', 'geoNetwork_metro', 'geoNetwork_city', 'geoNetwork_networkDomain',
                  'trafficSource_campaign', 'trafficSource_source', 'trafficSource_medium', 'trafficSource_keyword', 'trafficSource_adwordsClickInfo_isVideoAd',
                  'trafficSource_referralPath', 'trafficSource_isTrueDirect', 'trafficSource_adContent']
    
    # Separate numeric and factor columns
    print(group_var + [col for col in df.columns if col not in df_factors])
    print(group_var + df_factors)
    
    num_df = df.select(*group_var, *[col for col in df.columns if col not in df_factors])
    fac_df = df.select(*group_var, *df_factors)
    
    # Aggregating numeric features
    if len(num_df.columns) > 1:
        agg_exprs = []
        for col in num_df.columns:
            agg_exprs += [getattr(F, stat)(col).alias(f"{col}_{stat.upper()}") for stat in num_stats]
        num_df = num_df.groupBy(group_var).agg(*agg_exprs)
    
    # Aggregating factor features using mode
    if len(fac_df.columns) > 1:
        window_spec = Window.partitionBy(*group_var)
        fac_df = fac_df.withColumn('rank', F.row_number().over(window_spec.orderBy(*[F.col(g) for g in group_var])))
        fac_df = fac_df.filter(F.col('rank') == 1).drop('rank')
    
    # Merging numeric and factor DataFrames
    if len(num_df.columns) > 1 and len(fac_df.columns) > 1:
        agg_df = num_df.join(fac_df, group_var, "outer")
    elif len(num_df.columns) > 1:
        agg_df = num_df
    else:
        agg_df = fac_df
    
    # Renaming columns if a label is provided
    if label is not None:
        for col_name in agg_df.columns:
            if col_name not in group_var:
                agg_df = agg_df.withColumnRenamed(col_name, f"{label}_{col_name}")
    
    # Impute zeros for standard deviation columns
    if sd_zeros:
        for col_name in agg_df.columns:
            if "_std" in col_name:
                agg_df = agg_df.withColumn(col_name, F.coalesce(F.col(col_name), F.lit(0)))
    
    return agg_df
