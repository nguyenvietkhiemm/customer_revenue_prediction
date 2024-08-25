from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, BooleanType, ArrayType, MapType
from pyspark.sql.functions import col, from_json, element_at
import os, pandas as pd, ast, json
from pyspark.sql import functions as F

### IMPORT CSV WITH JSON

def infer_schema(sample):
    if isinstance(sample, list):
        if len(sample) > 0:
            return ArrayType(infer_schema(sample[0]))
        else:
            return ArrayType(StringType())
    elif isinstance(sample, dict):
        if len(sample) > 0:
            return StructType([StructField(k, infer_schema(v), True) for k, v in sample.items()])
        else:
            return StructType([])
    elif isinstance(sample, bool):
        return BooleanType()
    else:
        return StringType()
    
def merge_schemas(schema1, schema2):
    def merge_fields(fields1, fields2):
        fields1_dict = {f.name: f for f in fields1}
        for field in fields2:
            if field.name not in fields1_dict:
                fields1_dict[field.name] = field
            elif isinstance(field.dataType, StructType):
                # Nếu trường là StructType, hợp nhất các schema con
                existing_field = fields1_dict[field.name]
                merged_type = merge_schemas(existing_field.dataType, field.dataType)
                fields1_dict[field.name] = StructField(field.name, merged_type, field.nullable)
            elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
                # Nếu trường là ArrayType với StructType, hợp nhất các schema con của ArrayType
                existing_field = fields1_dict[field.name]
                merged_element_type = merge_schemas(existing_field.dataType.elementType, field.dataType.elementType)
                fields1_dict[field.name] = StructField(field.name, ArrayType(merged_element_type), field.nullable)
            elif isinstance(field.dataType, MapType):
                # MapType không cần hợp nhất vì tất cả MapType đều giống nhau
                fields1_dict[field.name] = StructField(field.name, field.dataType, field.nullable)

        return StructType(list(fields1_dict.values()))

    if isinstance(schema1, StructType) and isinstance(schema2, StructType):
        return StructType(merge_fields(schema1.fields, schema2.fields))
    elif isinstance(schema1, ArrayType) and isinstance(schema2, ArrayType):
        return ArrayType(merge_schemas(schema1.elementType, schema2.elementType))
    else:
        return schema1 if schema1 != StringType() else schema2
    
def handle_complex_schema(df, column, schema, prefix=""):
    if isinstance(schema, StructType):
        for field in schema.fields:
            field_name = field.name
            full_column_name = ".".join([prefix, column, field_name]) if prefix else ".".join([column, field_name])
            output_column_name = "_".join([prefix, column, field_name]) if prefix else "_".join([column, field_name])
            if isinstance(field.dataType, (StructType, ArrayType)):
                df = handle_complex_schema(df, field_name, field.dataType, ".".join([prefix, column]) if prefix else column)
            else:
                df = df.withColumn(output_column_name, col(full_column_name))
    
    elif isinstance(schema, ArrayType):
        element_type = schema.elementType
        if isinstance(element_type, StructType):
            for field in element_type.fields:
                field_name = field.name
                output_column_name = "_".join([prefix, column, field_name]) if prefix else "_".join([column, field_name])
                full_column_name = ".".join([prefix, column, field_name]) if prefix else ".".join([column, field_name])
            
                df = df.withColumn(output_column_name, element_at(col(full_column_name), 1))
    
    return df

def json_to_col(df, columns):
    for column in columns:
        try:
            row = json.loads(df.select(column).dropna().first()[0])
        except:
            row = ast.literal_eval(df.select(column).dropna().first()[0])
        schema = infer_schema(row)
        for row in df.select(column).dropna().limit(10000).collect():
            try:
                row = json.loads(row[0])
            except:
                row = ast.literal_eval(row[0])
            row_schema = infer_schema(row)
            schema = merge_schemas(schema, row_schema)
        print(schema)
        df = df.withColumn(column, from_json(col(column), schema))
        df = handle_complex_schema(df, column, schema)
    return df.drop(*columns)

### FILL NA

def fill_na(df):
    to_NA_cols = ['trafficSource_adContent',
                  'trafficSource_adwordsClickInfo_adNetworkType',
                  'trafficSource_adwordsClickInfo_slot',
                  'trafficSource_adwordsClickInfo_gclId',
                  'trafficSource_keyword',
                  'trafficSource_referralPath',
                  'customDimensions_value']

    cols_to_replace = {
            'device_browserSize' : 'not available in demo dataset', 
            'device_flashVersion' : 'not available in demo dataset', 
            'device_browserVersion' : 'not available in demo dataset', 
            'device_language' : 'not available in demo dataset',
            'device_mobileDeviceBranding' : 'not available in demo dataset',
            'device_mobileDeviceInfo' : 'not available in demo dataset',
            'device_mobileDeviceMarketingName' : 'not available in demo dataset',
            'device_mobileDeviceModel' : 'not available in demo dataset',
            'device_mobileInputSelector' : 'not available in demo dataset',
            'device_operatingSystemVersion' : 'not available in demo dataset',
            'device_screenColors' : 'not available in demo dataset',
            'device_screenResolution' : 'not available in demo dataset',
            'geoNetwork_city' : 'not available in demo dataset',
            'geoNetwork_cityId' : 'not available in demo dataset',
            'geoNetwork_latitude' : 'not available in demo dataset',
            'geoNetwork_longitude' : 'not available in demo dataset',
            'geoNetwork_metro' : ['not available in demo dataset', '(not set)'], 
            'geoNetwork_networkDomain' : ['unknown.unknown', '(not set)'], 
            'geoNetwork_networkLocation' : 'not available in demo dataset',
            'geoNetwork_region' : 'not available in demo dataset',
            'trafficSource_adwordsClickInfo_criteriaParameters' : 'not available in demo dataset',
            'trafficSource_medium': '(none)',
            'trafficSource_campaign' : '(not set)', 
            'trafficSource_keyword' : ['(not provided)', '(not set)'], 
            'geoNetwork_networkDomain': '(not set)', 
            'geoNetwork_city': ['not available in demo dataset', '(not set)'],
            
        }

    to_0_columns = ['totals_transactionRevenue',
                    'trafficSource_adwordsClickInfo_page',
                    'totals_sessionQualityDim',
                    'totals_bounces',
                    'totals_timeOnSite',
                    'totals_newVisits',
                    'totals_pageviews',
                    'customDimensions_index',
                    'totals_transactions',
                    'totals_totalTransactionRevenue']

    to_true_cols  = ['trafficSource_adwordsClickInfo_isVideoAd']
    to_false_cols = ['trafficSource_isTrueDirect'] 
    
    # convert to integers
    to_int = ['customDimensions_index',
            'totals_bounces',
            'totals_newVisits',
            'totals_pageviews',
            'totals_hits',
            'totals_sessionQualityDim',
            'totals_visits',
            'totals_timeOnSite',
            'trafficSource_adwordsClickInfo_page',
            'totals_transactions',
            'totals_transactionRevenue',
            'totals_totalTransactionRevenue']
    
    for col_name, values_to_replace in cols_to_replace.items():
        df = df.replace(to_replace=values_to_replace,
                        value="NA", subset=col_name)
    df = df.fillna("NA", subset=to_NA_cols)
    df = df.fillna("0", subset=to_0_columns)
    df = df.fillna(True, to_true_cols)
    df = df.fillna(False, to_false_cols)
    for col_name in to_int :
        df = df.withColumn(col_name, col(col_name).cast(IntegerType()))
    df = df.withColumn("date", F.to_date(F.col("date"), "yyyyMMdd"))
    return df

def drop_single_value_columns(df):
    unique_counts = df.agg(*(F.countDistinct(col).alias(col) for col in df.columns)).collect()[0]
    columns_to_drop = [col for col, count in zip(df.columns, unique_counts) if count == 1]
    print(columns_to_drop)
    return df.drop(*columns_to_drop)