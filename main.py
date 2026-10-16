from parsers.parse_json import parse_json, user_info_json
from parsers.utils import drop_empty_columns, drop_columns
from parsers.parse_tab import parse_tab
from parsers.parse_yaml import parse_yaml
json_path = r'C:\Users\Vladislav\Desktop\Airlines\FrequentFlyerForum-Profiles.json'
parquet_json_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_json.parquet'


tab_path = r'C:\Users\Vladislav\Desktop\Airlines\Sirena-export-fixed.tab'
parquet_tab_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_tab.parquet'

yaml_path = r'C:\Users\Vladislav\Desktop\Airlines\SkyTeam-Exchange.yaml'
parquet_yaml_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_yaml.parquet'

df = parse_json(json_path, parquet_json_path)
df = drop_empty_columns(df,['passports'],output_path=parquet_json_path)
df = drop_columns(df,['sex'],output_path=parquet_json_path)
# pt = parse_tab(tab_path,parquet_tab_path)
# pt = drop_empty_columns(pt,['Seat'],output_path=parquet_tab_path)

# df = parse_yaml(yaml_path,parquet_yaml_path)
# df = drop_empty_columns(df,['status'],output_path=parquet_yaml_path)