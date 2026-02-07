from src.data.bad_data_generator import BadDataGenerator
from src.profilers.spark_profiler import SparkDataProfiler

generator = BadDataGenerator()
profiler = SparkDataProfiler()

# Generate bad data
clean_df = generator.generate_clean_dataset(n_rows=1000)
_, bad_df = generator.generate_quality_issue_dataset(
    n_rows=1000,
    issue_type='outlier',
    outlier_rate=0.15
)

# Profile it
profile = profiler.profile_dataset(bad_df, baseline_df=clean_df)

# Show me ALL the metric names and their values
print("PROFILER OUTPUT METRICS:")
for key, value in profile.items():
    print(f"  {key}: {value}")