"""
国际足球队名称 - 中文映射表
覆盖：FIFA 211个会员协会 + 历史名称 + 特殊球队
"""

# 主要国家队英中映射
TEAM_ZH = {
    # 亚洲 AFC (47)
    "Japan": "日本", "South Korea": "韩国", "Iran": "伊朗", "Saudi Arabia": "沙特阿拉伯",
    "Australia": "澳大利亚", "Qatar": "卡塔尔", "United Arab Emirates": "阿联酋",
    "Iraq": "伊拉克", "United Arab Republic": "阿联酋(历史)", "Oman": "阿曼",
    "Uzbekistan": "乌兹别克斯坦", "Syria": "叙利亚", "Jordan": "约旦",
    "Vietnam": "越南", "Vietnam Republic": "越南共和国(历史)", "Lebanon": "黎巴嫩",
    "Palestine": "巴勒斯坦", "Mandatory Palestine": "巴勒斯坦(托管地)",
    "Kyrgyzstan": "吉尔吉斯斯坦", "India": "印度", "Tajikistan": "塔吉克斯坦",
    "Thailand": "泰国", "China PR": "中国", "China": "中国",
    "Chinese Taipei": "中华台北", "Taiwan": "中国台湾", "Hong Kong": "中国香港",
    "Macau": "中国澳门", "North Korea": "朝鲜", "Philippines": "菲律宾",
    "Bahrain": "巴林", "Kuwait": "科威特", "Malaysia": "马来西亚",
    "Singapore": "新加坡", "Indonesia": "印度尼西亚", "Dutch East Indies": "荷属东印度(历史)",
    "Myanmar": "缅甸", "Burma": "缅甸(历史)", "Cambodia": "柬埔寨",
    "Laos": "老挝", "Brunei": "文莱", "Bangladesh": "孟加拉国",
    "Maldives": "马尔代夫", "Nepal": "尼泊尔", "Sri Lanka": "斯里兰卡",
    "Mongolia": "蒙古", "Afghanistan": "阿富汗", "Pakistan": "巴基斯坦",
    "Yemen": "也门", "Yemen Arab Republic": "阿拉伯也门共和国(历史)",
    "South Yemen": "南也门(历史)", "Timor-Leste": "东帝汶", "Guam": "关岛",

    # 非洲 CAF (54)
    "Senegal": "塞内加尔", "Morocco": "摩洛哥", "Nigeria": "尼日利亚",
    "Egypt": "埃及", "Algeria": "阿尔及利亚", "Tunisia": "突尼斯",
    "Cameroon": "喀麦隆", "Ghana": "加纳", "Côte d'Ivoire": "科特迪瓦",
    "Ivory Coast": "科特迪瓦", "South Africa": "南非", "Mali": "马里",
    "DR Congo": "刚果(金)", "Congo-Kinshasa": "刚果(金)(历史)", "Zaïre": "扎伊尔(历史)",
    "Congo-Léopoldville": "刚果-利奥波德维尔(历史)", "Belgian Congo": "比属刚果(历史)",
    "Congo": "刚果(布)", "Congo-Brazzaville": "刚果(布)", "Burkina Faso": "布基纳法索",
    "Upper Volta": "上沃尔特(历史)", "Cape Verde": "佛得角",
    "Guinea": "几内亚", "Zambia": "赞比亚", "Gabon": "加蓬",
    "Benin": "贝宁", "Dahomey": "达荷美(历史)", "Togo": "多哥",
    "Kenya": "肯尼亚", "Madagascar": "马达加斯加", "Uganda": "乌干达",
    "Zimbabwe": "津巴布韦", "Southern Rhodesia": "南罗得西亚(历史)",
    "Rhodesia": "罗得西亚(历史)", "Angola": "安哥拉", "Mozambique": "莫桑比克",
    "Tanzania": "坦桑尼亚", "Rwanda": "卢旺达", "Burundi": "布隆迪",
    "Ethiopia": "埃塞俄比亚", "Sudan": "苏丹", "South Sudan": "南苏丹",
    "Libya": "利比亚", "Liberia": "利比里亚", "Sierra Leone": "塞拉利昂",
    "Mauritania": "毛里塔尼亚", "Gambia": "冈比亚", "Gambia, The": "冈比亚",
    "Botswana": "博茨瓦纳", "Bechuanaland": "贝专纳(历史)", "Eswatini": "斯威士兰",
    "Swaziland": "斯威士兰(历史)", "Lesotho": "莱索托", "Basutoland": "巴苏陀兰(历史)",
    "Malawi": "马拉维", "Nyasaland": "尼亚萨兰(历史)", "Namibia": "纳米比亚",
    "South West Africa": "西南非洲(历史)", "Comoros": "科摩罗",
    "Mauritius": "毛里求斯", "Seychelles": "塞舌尔", "Djibouti": "吉布提",
    "French Somaliland": "法属索马里(历史)", "Equatorial Guinea": "赤道几内亚",
    "São Tomé and Príncipe": "圣多美和普林西比", "Central African Republic": "中非共和国",
    "Chad": "乍得", "Niger": "尼日尔", "Somalia": "索马里",
    "Réunion": "留尼汪", "Zanzibar": "桑给巴尔",

    # 中北美 CONCACAF (41)
    "United States": "美国", "USA": "美国", "Mexico": "墨西哥",
    "Costa Rica": "哥斯达黎加", "Canada": "加拿大", "Panama": "巴拿马",
    "Honduras": "洪都拉斯", "Jamaica": "牙买加", "El Salvador": "萨尔瓦多",
    "Guatemala": "危地马拉", "Curaçao": "库拉索", "Netherlands Antilles": "荷属安的列斯(历史)",
    "Trinidad and Tobago": "特立尼达和多巴哥", "Haiti": "海地",
    "Cuba": "古巴", "Bermuda": "百慕大", "Nicaragua": "尼加拉瓜",
    "Dominican Republic": "多米尼加共和国", "Barbados": "巴巴多斯",
    "Bahamas": "巴哈马", "Suriname": "苏里南", "Guyana": "圭亚那",
    "British Guiana": "英属圭亚那(历史)", "Puerto Rico": "波多黎各",
    "Saint Kitts and Nevis": "圣基茨和尼维斯", "Saint Lucia": "圣卢西亚",
    "Saint Vincent and the Grenadines": "圣文森特和格林纳丁斯",
    "Grenada": "格林纳达", "Antigua and Barbuda": "安提瓜和巴布达",
    "Dominica": "多米尼克", "Aruba": "阿鲁巴", "Cayman Islands": "开曼群岛",
    "Montserrat": "蒙特塞拉特", "Turks and Caicos Islands": "特克斯和凯科斯群岛",
    "British Virgin Islands": "英属维尔京群岛", "U.S. Virgin Islands": "美属维尔京群岛",
    "Saint Martin": "圣马丁", "Sint Maarten": "圣马丁(荷)",
    "Anguilla": "安圭拉", "Greenland": "格陵兰", "Belize": "伯利兹",

    # 南美 CONMEBOL (10)
    "Brazil": "巴西", "Argentina": "阿根廷", "Uruguay": "乌拉圭",
    "Colombia": "哥伦比亚", "Chile": "智利", "Peru": "秘鲁",
    "Ecuador": "厄瓜多尔", "Paraguay": "巴拉圭", "Bolivia": "玻利维亚",
    "Venezuela": "委内瑞拉", "Guiana": "圭亚那(法)", "French Guiana": "法属圭亚那",

    # 欧洲 UEFA (55)
    "Germany": "德国", "Germany DR": "民主德国(历史)", "West Germany": "西德(历史)",
    "Germany United": "德国统一队(历史)", "Saarland": "萨尔保护领(历史)",
    "France": "法国", "Spain": "西班牙", "England": "英格兰",
    "Italy": "意大利", "Portugal": "葡萄牙", "Netherlands": "荷兰",
    "Belgium": "比利时", "Croatia": "克罗地亚", "Serbia": "塞尔维亚",
    "Serbia and Montenegro": "塞尔维亚和黑山(历史)", "Yugoslavia": "南斯拉夫(历史)",
    "SFR Yugoslavia": "南斯拉夫(历史)", "FR Yugoslavia": "南斯拉夫联盟(历史)",
    "Poland": "波兰", "Ukraine": "乌克兰", "Russia": "俄罗斯",
    "Soviet Union": "苏联(历史)", "CIS": "独联体(历史)",
    "Austria": "奥地利", "Switzerland": "瑞士", "Denmark": "丹麦",
    "Sweden": "瑞典", "Norway": "挪威", "Finland": "芬兰",
    "Iceland": "冰岛", "Ireland": "爱尔兰", "Republic of Ireland": "爱尔兰",
    "Northern Ireland": "北爱尔兰", "Scotland": "苏格兰", "Wales": "威尔士",
    "Czech Republic": "捷克", "Czechia": "捷克", "Czechoslovakia": "捷克斯洛伐克(历史)",
    "Bohemia": "波希米亚(历史)", "Bohemia and Moravia": "波希米亚和摩拉维亚(历史)",
    "Representation of Czechs and Slovaks": "捷克和斯洛伐克代表队(历史)",
    "Slovakia": "斯洛伐克", "Hungary": "匈牙利", "Romania": "罗马尼亚",
    "Bulgaria": "保加利亚", "Greece": "希腊", "Turkey": "土耳其",
    "Albania": "阿尔巴尼亚", "Armenia": "亚美尼亚", "Azerbaijan": "阿塞拜疆",
    "Belarus": "白俄罗斯", "Bosnia and Herzegovina": "波黑",
    "Cyprus": "塞浦路斯", "Estonia": "爱沙尼亚", "Faroe Islands": "法罗群岛",
    "Georgia": "格鲁吉亚", "Kazakhstan": "哈萨克斯坦", "Kosovo": "科索沃",
    "Latvia": "拉脱维亚", "Liechtenstein": "列支敦士登", "Lithuania": "立陶宛",
    "Luxembourg": "卢森堡", "Malta": "马耳他", "Moldova": "摩尔多瓦",
    "Monaco": "摩纳哥", "Montenegro": "黑山", "North Macedonia": "北马其顿",
    "Macedonia": "马其顿(历史)", "North Macedonia": "北马其顿", "San Marino": "圣马力诺",
    "Andorra": "安道尔", "Gibraltar": "直布罗陀", "Vatican City": "梵蒂冈",
    "Israel": "以色列",

    # 大洋洲 OFC (13)
    "New Zealand": "新西兰", "Fiji": "斐济", "Papua New Guinea": "巴布亚新几内亚",
    "Solomon Islands": "所罗门群岛", "Vanuatu": "瓦努阿图", "New Caledonia": "新喀里多尼亚",
    "Tahiti": "塔希提", "Samoa": "萨摩亚", "Western Samoa": "西萨摩亚(历史)",
    "American Samoa": "美属萨摩亚", "Tonga": "汤加", "Cook Islands": "库克群岛",
    "Tuvalu": "图瓦卢", "Micronesia": "密克罗尼西亚", "Northern Mariana Islands": "北马里亚纳群岛",

    # 其他/特殊
    "Korea Republic": "韩国", "Korea DPR": "朝鲜", "IR Iran": "伊朗",
    "Côte d'Ivoire": "科特迪瓦", "Bosnia-Herzegovina": "波黑",
    "China PR": "中国", "Republic of Ireland": "爱尔兰",
    "Yugoslavia": "南斯拉夫(历史)", "Saar": "萨尔(历史)",
    "Vietnam Republic": "南越(历史)",
    "Czechoslovakia": "捷克斯洛伐克(历史)",
    "Newfoundland": "纽芬兰(历史)",
    "Brittany": "布列塔尼(非官方)",
    "Catalonia": "加泰罗尼亚(非官方)",
    "Basque Country": "巴斯克地区(非官方)",
    "Galicia": "加利西亚(非官方)",
    "Sápmi": "萨米(非官方)",
    "Kernow": "康沃尔(非官方)",
    "Isle of Man": "马恩岛(非官方)",
    "Jersey": "泽西岛(非官方)",
    "Guernsey": "根西岛(非官方)",
    "Aldous Huxley": "奥尔杜斯·赫胥黎",
    "Two Sicilies": "两西西里(历史)",
    "Mayotte": "马约特",
    "Saint Pierre and Miquelon": "圣皮埃尔和密克隆",
    "Wallis and Futuna": "瓦利斯和富图纳",
    "Kiribati": "基里巴斯",
    "Nauru": "瑙鲁",
    "Palau": "帕劳",
    "Marshall Islands": "马绍尔群岛",
    "Federal Republic of Central America": "中美洲联邦共和国(历史)",
    "Manchuria": "满洲国(历史)",
    "Mukden": "沈阳(历史)",
    "Bohemia": "波希米亚(历史)",
}


def get_chinese_name(team_en):
    """获取中文名，找不到则返回原名"""
    return TEAM_ZH.get(team_en, team_en)


def get_all_teams_with_chinese(teams):
    """获取所有球队及其对应中文名"""
    return {t: get_chinese_name(t) for t in sorted(teams)}


if __name__ == "__main__":
    # 测试
    test_teams = ["Brazil", "Germany", "Argentina", "China PR", "Yugoslavia", "Soviet Union"]
    for t in test_teams:
        print(f"{t:25} -> {get_chinese_name(t)}")
