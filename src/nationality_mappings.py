def get_mappings():

    nationality2country = {
        'Belgian': 'Belgium',
        'British (English)': 'United Kingdom',
        'Canadian': 'Canada',
        'American': 'United States of America',
        'Slovakian': 'Slovakia',
        'Dutch': 'Netherlands',
        'Spanish': 'Spain',
        'Indian': 'India',
        'German': 'Germany',
        'Russian': 'Russia',
        'Chinese': 'China',
        'Polish': 'Poland',
        'Croatian': 'Croatia',
        'Brazilian': 'Brazil',
        'Danish': 'Denmark',
        'Jamaican': 'Jamaica',
        'Hungarian': 'Hungary',
        'Swedish': 'Sweden',
        'Serbian': 'Serbia',
        'Senegalese': 'Senegal',
        'Czech': 'Czechia',
        'Mexican': 'Mexico',
        'Cuban': 'Cuba',
        'French': 'France',
        'Japanese': 'Japan',
        'Australian': 'Australia',
        'Sudanese': 'Sudan',
        'Belarusian': 'Belarus',
        'Norwegian': 'Norway',
        'Thai': 'Thailand',
        'Somalian': 'Somalia',
        'Lithuanian': 'Lithuania',
        'Greek': 'Greece',
        'Angolan': 'Angola',
        'Ukrainian': 'Ukraine',
        'Latvian': 'Latvia',
        'Indonesian': 'Indonesia',
        'Romanian': 'Romania',
        'Estonian': 'Estonia',
        'Bahamian': 'Bahamas',
        'Argentinian': 'Argentina',
        'Nigerian': 'Nigeria',
        'Georgian': 'Georgia',
        'Uzbekistani': 'Uzbekistan',
        'British': 'United Kingdom',
        'Dominican': 'Dominican Rep.',
        'Israeli': 'Israel',
        'Irish': 'Ireland',
        'Malaysian': 'Malaysia',
        'Peruvian': 'Peru',
        'South Korean': 'South Korea',
        'Portuguese': 'Portugal',
        'Singaporean': 'Singapore',
        'Venezuelan': 'Venezuela',
        'New Zealander': 'New Zealand',
        'Italian': 'Italy',
        'Finnish': 'Finland',
        'Ghanaian': 'Ghana',
        'Bulgarian': 'Bulgaria',
        'Austrian': 'Austria',
        'South African': 'South Africa',
        'Swiss': 'Switzerland',
        'Malian': 'Mali',
        'Slovenian': 'Slovenia',
        'Ivorian': "Côte d'Ivoire",
        'Burundian': 'Burundi',
        'Algerian': 'Algeria',
        'Kenyan': 'Kenya',
        'South Sudanese': 'S. Sudan',
        'British (Scottish)': 'United Kingdom',
        'Colombian': 'Colombia',
        'Filipino': 'Philippines',
        'Puerto Rican': 'Puerto Rico',
        'Nepali': 'Nepal',
        'Panamanian': 'Panama',
        'Turkish': 'Turkey',
        'Chilean': 'Chile',
        'Montenegrin': 'Montenegro',
        'Costa Rican': 'Costa Rica',
        'Barbadian': 'Barbados',  # Not present in second list — closest match not found, may need to omit or add manually
        'Luxembourger': 'Luxembourg',
        'Kazakhstani': 'Kazakhstan',
        'Cape Verdean': 'Cape Verde',  # Not in 2nd list — could map to 'Cabo Verde' if included
        'Egyptian': 'Egypt',
        'Icelandic': 'Iceland',
        'Maltese': 'Malta',  # Not in 2nd list — may need to add
        'Taiwanese': 'Taiwan',
        'Uruguayan': 'Uruguay',
        'Moldovan': 'Moldova',
        'Bosnian': 'Bosnia and Herz.',
        'Zimbabwean': 'Zimbabwe',
        'Congolese': 'Congo',
        'Vietnamese': 'Vietnam',
        'Tunisian': 'Tunisia',
        'Guinean': 'Guinea',
        'Namibian': 'Namibia',
        'Ugandan': 'Uganda',
        'Honduran': 'Honduras',
        'Armenian': 'Armenia',
        'Guyanese': 'Guyana',
        'Guatemalan': 'Guatemala',
        'Rwandan': 'Rwanda',
        'British (Welsh)': 'United Kingdom',
        'Afghan': 'Afghanistan',
        'Azerbaijani': 'Azerbaijan',
        'Paraguayan': 'Paraguay',
        'Moroccan': 'Morocco',
        'Caymanian': 'Cayman Islands',  # Not in second list — may need to add manually
        'Ethiopian': 'Ethiopia',
        'Botswanan': 'Botswana',
        'Mauritian': 'Mauritius',  # Not in second list — may need to add
        'Trinidadian': 'Trinidad and Tobago',
        'Macedonian': 'North Macedonia',
        'Antiguan': 'Antigua and Barbuda',  # Not in second list — may need to add
        'Kyrgyzstani': 'Kyrgyzstan',
        'Albanian': 'Albania',
        'Gabonese': 'Gabon',
        'British (Northern Irish)': 'United Kingdom',
        'Tanzanian': 'Tanzania',
        'Salvadoran': 'El Salvador',
        'Aruban': 'Aruba',  # Not in second list — may need to add
        'Zambian': 'Zambia',
        'Martinican': 'Martinique',  # Not in second list — may need to add
        'Haitian': 'Haiti',
        'Gambian': 'Gambia',
        'Burkinabe': 'Burkina Faso',
        'Bermudian': 'Bermuda',  # Not in second list — may need to add
        'Kosovar': 'Kosovo',
        'Mongolian': 'Mongolia',
        'Sierra Leonean': 'Sierra Leone',
        'Iranian': 'Iran'
    }


    nationality_to_global = {
        # 🌍 Global North
        'American': 'Global North',
        'Canadian': 'Global North',
        'British': 'Global North',
        'British (English)': 'Global North',
        'British (Northern Irish)': 'Global North',
        'British (Scottish)': 'Global North',
        'British (Welsh)': 'Global North',
        'German': 'Global North',
        'French': 'Global North',
        'Dutch': 'Global North',
        'Belgian': 'Global North',
        'Swiss': 'Global North',
        'Austrian': 'Global North',
        'Irish': 'Global North',
        'Luxembourger': 'Global North',
        'Swedish': 'Global North',
        'Danish': 'Global North',
        'Norwegian': 'Global North',
        'Finnish': 'Global North',
        'Icelandic': 'Global North',
        'Italian': 'Global North',
        'Spanish': 'Global North',
        'Portuguese': 'Global North',
        'Greek': 'Global North',
        'Cypriot': 'Global North',
        'Maltese': 'Global North',
        'Australian': 'Global North',
        'New Zealander': 'Global North',

        # 🌎 Global South
        # Latin America and Caribbean
        'Mexican': 'Global South',
        'Brazilian': 'Global South',
        'Argentinian': 'Global South',
        'Colombian': 'Global South',
        'Chilean': 'Global South',
        'Peruvian': 'Global South',
        'Venezuelan': 'Global South',
        'Uruguayan': 'Global South',
        'Guyanese': 'Global South',
        'Cuban': 'Global South',
        'Dominican': 'Global South',
        'Jamaican': 'Global South',
        'Puerto Rican': 'Global South',
        'Haitian': 'Global South',
        'Guatemalan': 'Global South',
        'Costa Rican': 'Global South',
        'Panamanian': 'Global South',
        'Trinidadian': 'Global South',
        'Honduran': 'Global South',
        'Bahamian': 'Global South',
        'Martinican': 'Global South',
        'Aruban': 'Global South',
        'Caymanian': 'Global South',
        'Curaçaoan': 'Global South',
        'Antiguan': 'Global South',
        'Bermudian': 'Global South',

        # Africa & Middle East
        'Egyptian': 'Global South',
        'Moroccan': 'Global South',
        'Algerian': 'Global South',
        'Tunisian': 'Global South',
        'Nigerian': 'Global South',
        'South African': 'Global South',
        'Senegalese': 'Global South',
        'Ghanaian': 'Global South',
        'Kenyan': 'Global South',
        'Somalian': 'Global South',
        'Sudanese': 'Global South',
        'South Sudanese': 'Global South',
        'Ethiopian': 'Global South',
        'Ugandan': 'Global South',
        'Angolan': 'Global South',
        'Mozambican': 'Global South',
        'Burundian': 'Global South',
        'Congolese': 'Global South',
        'Namibian': 'Global South',
        'Cape Verdean': 'Global South',
        'Ivorian': 'Global South',
        'Malian': 'Global South',
        'Rwandan': 'Global South',
        'Guinean': 'Global South',
        'Gambian': 'Global South',
        'Beninese': 'Global South',
        'Sierra Leonean': 'Global South',
        'Burkinabe': 'Global South',

        # Asia
        'Chinese': 'Global South',
        'Japanese': 'Global South',
        'South Korean': 'Global South',
        'Taiwanese': 'Global South',
        'Mongolian': 'Global South',
        'Indian': 'Global South',
        'Sri Lankan': 'Global South',
        'Nepali': 'Global South',
        'Afghan': 'Global South',
        'Thai': 'Global South',
        'Indonesian': 'Global South',
        'Malaysian': 'Global South',
        'Vietnamese': 'Global South',
        'Filipino': 'Global South',
        'Singaporean': 'Global South',
        'Burmese': 'Global South',
        'Uzbekistani': 'Global South',
        'Kazakhstani': 'Global South',

        # Middle East / Caucasus
        'Israeli': 'Global South',
        'Lebanese': 'Global South',
        'Turkish': 'Global South',
        'Syrian': 'Global South',
        'Armenian': 'Global South',
        'Georgian': 'Global South',

        # Pacific Islands
        'French Polynesian': 'Global South',

        #more
        'Slovakian': 'Global North',
        'Russian': 'Global North',
        'Polish': 'Global North',
        'Croatian': 'Global North',
        'Hungarian': 'Global North',
        'Serbian': 'Global North',
        'Czech': 'Global North',
        'Belarusian': 'Global North',
        'Lithuanian': 'Global North',
        'Ukrainian': 'Global North',
        'Latvian': 'Global North',
        'Romanian': 'Global North',
        'Estonian': 'Global North',
        'Bulgarian': 'Global North',
        'Slovenian': 'Global North',
        'Montenegrin': 'Global North',
        'Moldovan': 'Global North',
        'Bosnian': 'Global North',
        'Macedonian': 'Global North',
        'Albanian': 'Global North',
        'Kosovar': 'Global North',

        # Caribbean / Latin America → Global South
        'Barbadian': 'Global South',
        'Paraguayan': 'Global South',
        'Salvadoran': 'Global South',

        # Africa → Global South
        'Zimbabwean': 'Global South',
        'Botswanan': 'Global South',
        'Mauritian': 'Global South',
        'Gabonese': 'Global South',
        'Tanzanian': 'Global South',
        'Zambian': 'Global South',

        # Central Asia / Caucasus / Middle East → Global South
        'Azerbaijani': 'Global South',
        'Kyrgyzstani': 'Global South',
        'Iranian': 'Global South',
    }

    nationality_to_region = {
            # North America
            'American': 'North America',
            'Canadian': 'North America',
            'Mexican': 'North America',
            
            # Caribbean/Central America
            'Dominican': 'Caribbean/Central America',
            'Cuban': 'Caribbean/Central America',
            'Honduran': 'Caribbean/Central America',
            'Jamaican': 'Caribbean/Central America',
            'Bahamian': 'Caribbean/Central America',
            'Puerto Rican': 'Caribbean/Central America',
            'Haitian': 'Caribbean/Central America',
            'Guatemalan': 'Caribbean/Central America',
            'Costa Rican': 'Caribbean/Central America',
            'Panamanian': 'Caribbean/Central America',
            'Trinidadian': 'Caribbean/Central America',
            'Martinican': 'Caribbean/Central America',
            'Aruban': 'Caribbean/Central America',
            'Caymanian': 'Caribbean/Central America',
            'Curaçaoan': 'Caribbean/Central America',
            'Antiguan': 'Caribbean/Central America',
            'Bermudian': 'Caribbean/Central America',
            
            # South America
            'Brazilian': 'South America',
            'Argentinian': 'South America',
            'Colombian': 'South America',
            'Chilean': 'South America',
            'Peruvian': 'South America',
            'Venezuelan': 'South America',
            'Uruguayan': 'South America',
            'Guyanese': 'South America',
            
            # Western Europe
            'British': 'Western Europe',
            'British (English)': 'Western Europe',
            'British (Northern Irish)': 'Western Europe',
            'British (Scottish)': 'Western Europe',
            'British (Welsh)': 'Western Europe',
            'German': 'Western Europe',
            'French': 'Western Europe',
            'Dutch': 'Western Europe',
            'Belgian': 'Western Europe',
            'Swiss': 'Western Europe',
            'Austrian': 'Western Europe',
            'Irish': 'Western Europe',
            'Luxembourger': 'Western Europe',
            
            # Southern Europe
            'Spanish': 'Southern Europe',
            'Italian': 'Southern Europe',
            'Portuguese': 'Southern Europe',
            'Greek': 'Southern Europe',
            'Maltese': 'Southern Europe',
            'Cypriot': 'Southern Europe',
            
            # Northern Europe
            'Swedish': 'Northern Europe',
            'Danish': 'Northern Europe',
            'Norwegian': 'Northern Europe',
            'Finnish': 'Northern Europe',
            'Icelandic': 'Northern Europe',
            
            # Eastern Europe
            'Russian': 'Eastern Europe',
            'Polish': 'Eastern Europe',
            'Czech': 'Eastern Europe',
            'Slovakian': 'Eastern Europe',
            'Ukrainian': 'Eastern Europe',
            'Romanian': 'Eastern Europe',
            'Bulgarian': 'Eastern Europe',
            'Hungarian': 'Eastern Europe',
            'Croatian': 'Eastern Europe',
            'Serbian': 'Eastern Europe',
            'Slovenian': 'Eastern Europe',
            'Estonian': 'Eastern Europe',
            'Latvian': 'Eastern Europe',
            'Lithuanian': 'Eastern Europe',
            'Belarusian': 'Eastern Europe',
            'Montenegrin': 'Eastern Europe',
            'Moldovan': 'Eastern Europe',
            'Bosnian': 'Eastern Europe',
            'Albanian': 'Eastern Europe',
            'Macedonian': 'Eastern Europe',
            'Kosovar': 'Eastern Europe',
            
            # East Asia
            'Chinese': 'East Asia',
            'Japanese': 'East Asia',
            'South Korean': 'East Asia',
            'Taiwanese': 'East Asia',
            'Mongolian': 'East Asia',
            
            # Southeast Asia
            'Thai': 'Southeast Asia',
            'Indonesian': 'Southeast Asia',
            'Malaysian': 'Southeast Asia',
            'Singaporean': 'Southeast Asia',
            'Vietnamese': 'Southeast Asia',
            'Filipino': 'Southeast Asia',
            'Burmese': 'Southeast Asia',
            
            # South Asia
            'Indian': 'South Asia',
            'Sri Lankan': 'South Asia',
            'Nepali': 'South Asia',
            'Afghan': 'South Asia',
            
            # Middle East
            'Israeli': 'Middle East',
            'Lebanese': 'Middle East',
            'Turkish': 'Middle East',
            'Syrian': 'Middle East',
            'Armenian': 'Middle East',
            
            # Central Asia
            'Uzbekistani': 'Central Asia',
            'Kazakhstani': 'Central Asia',
            
            # Sub-Saharan Africa
            'Nigerian': 'Sub-Saharan Africa',
            'South African': 'Sub-Saharan Africa',
            'Senegalese': 'Sub-Saharan Africa',
            'Ghanaian': 'Sub-Saharan Africa',
            'Kenyan': 'Sub-Saharan Africa',
            'Somalian': 'Sub-Saharan Africa',
            'South Sudanese': 'Sub-Saharan Africa',
            'Sudanese': 'Sub-Saharan Africa',
            'Angolan': 'Sub-Saharan Africa',
            'Burundian': 'Sub-Saharan Africa',
            'Ivorian': 'Sub-Saharan Africa',
            'Malian': 'Sub-Saharan Africa',
            'Rwandan': 'Sub-Saharan Africa',
            'Mozambican': 'Sub-Saharan Africa',
            'Cape Verdean': 'Sub-Saharan Africa',
            'Ethiopian': 'Sub-Saharan Africa',
            'Ugandan': 'Sub-Saharan Africa',
            'Guinean': 'Sub-Saharan Africa',
            'Gambian': 'Sub-Saharan Africa',
            'Sierra Leonean': 'Sub-Saharan Africa',
            'Beninese': 'Sub-Saharan Africa',
            'Tanzanian': 'Sub-Saharan Africa',
            'Burkinabe': 'Sub-Saharan Africa',
            'Congolese': 'Sub-Saharan Africa',
            'Namibian': 'Sub-Saharan Africa',
            
            # North Africa
            'Moroccan': 'North Africa',
            'Algerian': 'North Africa',
            'Tunisian': 'North Africa',
            'Egyptian': 'North Africa',
            
            # Caucasus
            'Georgian': 'Caucasus',
            
            # Oceania
            'Australian': 'Oceania',
            'New Zealander': 'Oceania',
            'French Polynesian': 'Oceania',
        }

    country_to_superregion = {
        # Europe
        'United Kingdom': 'Europe',
        'France': 'Europe',
        'Germany': 'Europe',
        'Italy': 'Europe',
        'Spain': 'Europe',
        'Netherlands': 'Europe',
        'Belgium': 'Europe',
        'Sweden': 'Europe',
        'Denmark': 'Europe',
        'Norway': 'Europe',
        'Finland': 'Europe',
        'Austria': 'Europe',
        'Ireland': 'Europe',
        'Greece': 'Europe',
        'Portugal': 'Europe',
        'Switzerland': 'Europe',
        'Poland': 'Europe',
        'Hungary': 'Europe',
        'Czechia': 'Europe',
        'Slovakia': 'Europe',
        'Slovenia': 'Europe',
        'Croatia': 'Europe',
        'Lithuania': 'Europe',
        'Latvia': 'Europe',
        'Estonia': 'Europe',
        'Ukraine': 'Europe',
        'Belarus': 'Europe',
        'Romania': 'Europe',
        'Bulgaria': 'Europe',
        'Bosnia and Herz.': 'Europe',
        'North Macedonia': 'Europe',
        'Montenegro': 'Europe',
        'Serbia': 'Europe',
        'Moldova': 'Europe',
        'Albania': 'Europe',
        'Kosovo': 'Europe',
        'Iceland': 'Europe',
        'Luxembourg': 'Europe',
        'Malta': 'Europe',
        'Russia' : "Europe",

        # North America
        'United States of America': 'North America',
        'Canada': 'North America',

        # South and Latin America
        'Mexico': 'South and Latin America',
        'Brazil': 'South and Latin America',
        'Argentina': 'South and Latin America',
        'Chile': 'South and Latin America',
        'Peru': 'South and Latin America',
        'Colombia': 'South and Latin America',
        'Venezuela': 'South and Latin America',
        'Uruguay': 'South and Latin America',
        'Paraguay': 'South and Latin America',
        'Cuba': 'South and Latin America',
        'Dominican Rep.': 'South and Latin America',
        'Puerto Rico': 'South and Latin America',
        'Jamaica': 'South and Latin America',
        'Trinidad and Tobago': 'South and Latin America',
        'Honduras': 'South and Latin America',
        'Panama': 'South and Latin America',
        'Costa Rica': 'South and Latin America',
        'Guatemala': 'South and Latin America',
        'El Salvador': 'South and Latin America',
        'Barbados': 'South and Latin America',
        'Bahamas': 'South and Latin America',
        'Aruba': 'South and Latin America',
        'Antigua and Barbuda': 'South and Latin America',
        'Guyana': 'South and Latin America',
        'Martinique': 'South and Latin America',
        'Cayman Islands': 'South and Latin America',
        'Bermuda': 'South and Latin America',

        # Africa
        'Nigeria': 'Africa',
        'South Africa': 'Africa',
        'Kenya': 'Africa',
        'Ethiopia': 'Africa',
        'Ghana': 'Africa',
        'Senegal': 'Africa',
        'Sudan': 'Africa',
        'S. Sudan': 'Africa',
        'Algeria': 'Africa',
        'Morocco': 'Africa',
        'Tunisia': 'Africa',
        'Uganda': 'Africa',
        'Zimbabwe': 'Africa',
        'Zambia': 'Africa',
        'Namibia': 'Africa',
        'Mali': 'Africa',
        'Burundi': 'Africa',
        'Rwanda': 'Africa',
        'Guinea': 'Africa',
        'Gambia': 'Africa',
        "Côte d'Ivoire": 'Africa',
        'Mozambique': 'Africa',
        'Botswana': 'Africa',
        'Cape Verde': 'Africa',
        'Burkina Faso': 'Africa',
        'Gabon': 'Africa',
        'Mauritius': 'Africa',
        'Sierra Leone': 'Africa',
        'Congo': 'Africa',

        # Asia
        'China': 'Asia',
        'Japan': 'Asia',
        'India': 'Asia',
        'South Korea': 'Asia',
        'North Korea': 'Asia',
        'Thailand': 'Asia',
        'Vietnam': 'Asia',
        'Philippines': 'Asia',
        'Indonesia': 'Asia',
        'Malaysia': 'Asia',
        'Singapore': 'Asia',
        'Bangladesh': 'Asia',
        'Pakistan': 'Asia',
        'Nepal': 'Asia',
        'Sri Lanka': 'Asia',
        'Afghanistan': 'Asia',
        'Uzbekistan': 'Asia',
        'Kazakhstan': 'Asia',
        'Kyrgyzstan': 'Asia',
        'Israel': 'Asia',
        'Iran': 'Asia',
        'Iraq': 'Asia',
        'Turkey': 'Asia',
        'Georgia': 'Asia',
        'Armenia': 'Asia',
        'Taiwan': 'Asia',
        'Azerbaijan': 'Asia',

        # Oceania
        'Australia': 'Oceania',
        'New Zealand': 'Oceania',
        'French Polynesia': 'Oceania',
        
        #extra added later
        # Africa
        "Angola": "Africa",
        "Benin": "Africa",
        "Cameroon": "Africa",
        "Central African Rep.": "Africa",
        "Chad": "Africa",
        "Dem. Rep. Congo": "Africa",
        "Djibouti": "Africa",
        "Egypt": "Africa",
        "Eq. Guinea": "Africa",
        "Eritrea": "Africa",
        "Guinea-Bissau": "Africa",
        "Lesotho": "Africa",
        "Liberia": "Africa",
        "Libya": "Africa",
        "Madagascar": "Africa",
        "Malawi": "Africa",
        "Mauritania": "Africa",
        "Niger": "Africa",
        "Somalia": "Africa",
        "Somaliland": "Africa",
        "South Sudan": "Africa",  # (if relevant)
        "Tanzania": "Africa",
        "Togo": "Africa",
        "eSwatini": "Africa",
        "W. Sahara": "Africa",

        # Asia
        "Bhutan": "Asia",
        "Brunei": "Asia",
        "Cambodia": "Asia",
        "Laos": "Asia",
        "Lebanon": "Asia",
        "Myanmar": "Asia",
        "Mongolia": "Asia",
        "Oman": "Asia",
        "Palestine": "Asia",
        "Qatar": "Asia",
        "Saudi Arabia": "Asia",
        "Syria": "Asia",
        "Tajikistan": "Asia",
        "Timor-Leste": "Asia",
        "Turkmenistan": "Asia",
        "United Arab Emirates": "Asia",
        "Yemen": "Asia",
        "Jordan": "Asia",
        "Kuwait": "Asia",

        # Oceania
        "Fiji": "Oceania",
        "New Caledonia": "Oceania",
        "Papua New Guinea": "Oceania",
        "Solomon Is.": "Oceania",
        "Vanuatu": "Oceania",

        # Latin America
        "Belize": "South and Latin America",
        "Bolivia": "South and Latin America",
        "Ecuador": "South and Latin America",
        "Haiti": "South and Latin America",
        "Nicaragua": "South and Latin America",
        "Suriname": "South and Latin America",

        # Europe
        "Cyprus": "Europe",
        "N. Cyprus": "Europe",  # often handled differently, optional

        "Falkland Is.": "South and Latin America",  
        "Fr. S. Antarctic Lands": "Africa",  
        "Greenland": "Europe" }

    # Convert nationality_to_global → country_to_global
    country_to_global = {
        nationality2country[nationality]: global_region
        for nationality, global_region in nationality_to_global.items()
        if nationality in nationality2country
    }

    # Convert nationality_to_region → country_to_region
    country_to_region = {
        nationality2country[nationality]: region
        for nationality, region in nationality_to_region.items()
        if nationality in nationality2country
    }

    return nationality2country, country_to_global, country_to_region, country_to_superregion
    #return nationality2country, nationality_to_global, nationality_to_region