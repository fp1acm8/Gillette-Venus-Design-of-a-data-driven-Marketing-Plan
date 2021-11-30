#from selenium import webdriver
import folium


# def get_comune(latitude: float, longitude: float):
#     """ Retrieve the italian "comune" given the latitude and the longitude.

#     Parameters
#     ----------
#         latitude: float
#             latitude of the location
#         longitude: float
#             longitude of the location

#     Returns
#     -------
#         comune: str
#             "comune" of the location
#     """
#     options = webdriver.ChromeOptions()
#     options.add_argument('--headless')
#     options.add_argument('--no-sandbox')
#     options.add_argument('--disable-dev-shm-usage')

#     wd = webdriver.Chrome('chromedriver',options=options)
#     url = "https://www.google.it/maps/search/" + str(latitude) + "," + str(longitude)
#     wd.get(url)
#     page = wd.page_source
#     end = page.rfind('Italy')
#     raw = page[end-100:end][::-1]
#     start = end - raw.find("\"\\[")
#     addr = page[start:end][8:]
#     comune = addr.split(', ')[0]
#     return comune


def plot_map_clusters(dataframe):
    # Create the map
    m = folium.Map(location=[42.55, 12.71], tiles='cartodbpositron', zoom_start=6)
    palette = ['orange', 'forestgreen', 'slategrey']

    # Add points to the map
    for _, row in dataframe.iterrows():
        folium.Circle(radius=1000, location=[row['latitude'], row['longitude']], color=palette[row['cluster']]).add_to(m)

    return m