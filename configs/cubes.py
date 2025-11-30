"""
Cube schema definitions for the three experimental datasets.

From Section 7.1.1:
- SalesCube: M5 retail sales data
- ManufacturingCube: Production operations data
- AirQualityCube: Environmental monitoring data
"""

from navllm.cube.schema import CubeSchema, Dimension, Measure, Level, AggregateFunction


def create_sales_cube_schema() -> CubeSchema:
    """
    SalesCube schema from M5 Forecasting dataset.
    
    Dimensions:
    - Time: day → week → month → quarter → year → All
    - Product: item_id → category → department → All
    - Store: store_id → state → All
    
    Measures: units_sold, revenue
    """
    time_dim = Dimension(
        name="Time",
        table="dim_time",
        key_column="date_id",
        levels=[
            Level("day", "day", 0),
            Level("week", "week", 1),
            Level("month", "month", 2),
            Level("quarter", "quarter", 3),
            Level("year", "year", 4),
            Level("All", "all_time", 5),
        ]
    )
    
    product_dim = Dimension(
        name="Product",
        table="dim_product",
        key_column="item_id",
        levels=[
            Level("item", "item_id", 0),
            Level("category", "cat_id", 1),
            Level("department", "dept_id", 2),
            Level("All", "all_product", 3),
        ]
    )
    
    store_dim = Dimension(
        name="Store",
        table="dim_store",
        key_column="store_id",
        levels=[
            Level("store", "store_id", 0),
            Level("state", "state_id", 1),
            Level("All", "all_store", 2),
        ]
    )
    
    return CubeSchema(
        name="SalesCube",
        fact_table="fact_sales",
        dimensions=[time_dim, product_dim, store_dim],
        measures=[
            Measure("units_sold", "units", AggregateFunction.SUM),
            Measure("revenue", "revenue", AggregateFunction.SUM),
        ]
    )


def create_manufacturing_cube_schema() -> CubeSchema:
    """
    ManufacturingCube schema for production operations.
    
    Dimensions:
    - Time: shift → day → week → month → All
    - Line: machine → line → plant → All
    - Product: variant → family → category → All
    
    Measures: throughput, defect_count, defect_rate
    """
    time_dim = Dimension(
        name="Time",
        table="dim_time",
        key_column="time_id",
        levels=[
            Level("shift", "shift_id", 0),
            Level("day", "day", 1),
            Level("week", "week", 2),
            Level("month", "month", 3),
            Level("All", "all_time", 4),
        ]
    )
    
    line_dim = Dimension(
        name="Line",
        table="dim_line",
        key_column="machine_id",
        levels=[
            Level("machine", "machine_id", 0),
            Level("line", "line_id", 1),
            Level("plant", "plant_id", 2),
            Level("All", "all_line", 3),
        ]
    )
    
    product_dim = Dimension(
        name="Product",
        table="dim_product",
        key_column="variant_id",
        levels=[
            Level("variant", "variant_id", 0),
            Level("family", "family_id", 1),
            Level("category", "category_id", 2),
            Level("All", "all_product", 3),
        ]
    )
    
    return CubeSchema(
        name="ManufacturingCube",
        fact_table="fact_production",
        dimensions=[time_dim, line_dim, product_dim],
        measures=[
            Measure("throughput", "throughput", AggregateFunction.SUM),
            Measure("defect_count", "defects", AggregateFunction.SUM),
            Measure("defect_rate", "defect_rate", AggregateFunction.AVG),
        ]
    )


def create_air_quality_cube_schema() -> CubeSchema:
    """
    AirQualityCube schema for environmental monitoring.
    
    Dimensions:
    - Time: hour → day → week → month → season → All
    - Location: station → district → city → region → All
    - Pollutant: (flat) PM2.5, PM10, NO2, O3, etc.
    
    Measures: concentration, aqi
    """
    time_dim = Dimension(
        name="Time",
        table="dim_time",
        key_column="time_id",
        levels=[
            Level("hour", "hour", 0),
            Level("day", "day", 1),
            Level("week", "week", 2),
            Level("month", "month", 3),
            Level("season", "season", 4),
            Level("All", "all_time", 5),
        ]
    )
    
    location_dim = Dimension(
        name="Location",
        table="dim_location",
        key_column="station_id",
        levels=[
            Level("station", "station_id", 0),
            Level("district", "district_id", 1),
            Level("city", "city_id", 2),
            Level("region", "region_id", 3),
            Level("All", "all_location", 4),
        ]
    )
    
    pollutant_dim = Dimension(
        name="Pollutant",
        table="dim_pollutant",
        key_column="pollutant_id",
        levels=[
            Level("pollutant", "pollutant_id", 0),
            Level("All", "all_pollutant", 1),
        ]
    )
    
    return CubeSchema(
        name="AirQualityCube",
        fact_table="fact_air_quality",
        dimensions=[time_dim, location_dim, pollutant_dim],
        measures=[
            Measure("concentration", "concentration", AggregateFunction.AVG),
            Measure("aqi", "aqi", AggregateFunction.AVG),
            Measure("max_concentration", "concentration", AggregateFunction.MAX),
            Measure("readings", "reading_id", AggregateFunction.COUNT),
        ]
    )


# Schema registry
CUBE_SCHEMAS = {
    "sales": create_sales_cube_schema,
    "manufacturing": create_manufacturing_cube_schema,
    "air_quality": create_air_quality_cube_schema,
}


def get_schema(name: str) -> CubeSchema:
    """Get cube schema by name."""
    if name not in CUBE_SCHEMAS:
        raise ValueError(f"Unknown cube: {name}. Available: {list(CUBE_SCHEMAS.keys())}")
    return CUBE_SCHEMAS[name]()
