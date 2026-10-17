from parsers.parse_json import parse_json, user_info_json
from parsers.utils import drop_empty_columns, drop_columns
from parsers.parse_tab import parse_tab
from parsers.parse_yaml import parse_yaml
from parsers.unify import build_unified_smart
import polars as pl

json_path = r'C:\Users\Vladislav\Desktop\Airlines\FrequentFlyerForum-Profiles.json'
parquet_json_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_json.parquet'


tab_path = r'C:\Users\Vladislav\Desktop\Airlines\Sirena-export-fixed.tab'
parquet_tab_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_tab.parquet'

yaml_path = r'C:\Users\Vladislav\Desktop\Airlines\SkyTeam-Exchange.yaml'
parquet_yaml_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_yaml.parquet'

df_json = parse_json(json_path, parquet_json_path)
df_tab = parse_tab(tab_path, parquet_tab_path)
df_yaml = parse_yaml(yaml_path, parquet_yaml_path)

df_json = drop_empty_columns(df_json,['passports'],output_path=parquet_json_path)
df_tab = drop_empty_columns(df_tab,['Seat'],output_path=parquet_yaml_path)
df_yaml = drop_empty_columns(df_yaml,['status'],output_path=parquet_yaml_path)

df_json = drop_columns(df_json,['sex'], output_path=parquet_json_path)
df_tab = drop_columns(df_tab,['Meal','Baggage','PaxAdditionalInfo'], output_path=parquet_tab_path)
#df_yaml= drop_columns()
output_path = r"C:\Users\Vladislav\Desktop\Airlines\unified.parquet"
unified_df = build_unified_smart(df_tab, df_yaml, df_json, output_path=output_path)

