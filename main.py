from parsers.parse_json import parse_json, user_info_json
from parsers.utils import drop_empty_columns, drop_columns
from parsers.parse_tab import parse_tab
from parsers.parse_yaml import parse_yaml
from parsers.parse_xml import parse_xml
from parsers.parse_csv import parse_csv
from parsers.parse_pdf import parse_pdf
from parsers.parse_boarding_passes import parse_boardingPass_zip
import pandas as pd
json_path = r'C:\Users\Vladislav\Desktop\Airlines\FrequentFlyerForum-Profiles.json'
parquet_json_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_json.parquet'


tab_path = r'C:\Users\Vladislav\Desktop\Airlines\Sirena-export-fixed.tab'
parquet_tab_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_tab.parquet'

yaml_path = r'C:\Users\Vladislav\Desktop\Airlines\SkyTeam-Exchange.yaml'
parquet_yaml_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_yaml.parquet'

xml_path = r'C:\Users\Vladislav\Desktop\Airlines\PointzAggregator-AirlinesData.xml'
parquet_xml_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_xml.parquet'

csv_path = r'C:\Users\Vladislav\Desktop\Airlines\BoardingData.csv'
parquet_csv_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_csv.parquet'


pdf_path = r'C:\Users\Vladislav\Desktop\Airlines\Skyteam_Timetable.pdf'
parquet_pdf_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_pdf.parquet'

zip_path = r'C:\Users\Vladislav\Desktop\Airlines\YourBoardingPassDotAero.zip'
parquet_zip_path = r'C:\Users\Vladislav\PycharmProjects\Parser\process\clean_zip.parquet'

def main():
    #df_json = parse_json(json_path, parquet_json_path)
    #df_tab = parse_tab(tab_path, parquet_tab_path)
    #df_yaml = parse_yaml(yaml_path, parquet_yaml_path)
    #df_xml = parse_xml(xml_path, parquet_xml_path)
    # df_csv = parse_csv(csv_path, parquet_csv_path)
    #df_pdf = parse_pdf(pdf_path,parquet_pdf_path)
    df_zip = parse_boardingPass_zip(zip_path, parquet_zip_path)



if __name__ == "__main__":
    main()