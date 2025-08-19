import streamlit as st
import pandas as pd
import sklearn.metrics as skmetric
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import zipfile

@st.cache_data
def load_beers():
	'''
	Loads beer data into a dataframe
	'''
	zf = zipfile.ZipFile('beer_reviews.csv.zip') # having First.csv zipped file.
	df_beers_total = pd.read_csv(zf.open('beer_reviews.csv'))
	df_beers_total['brewery_name'] = df_beers_total['brewery_name'].astype(str)
	styles = []
	for style in df_beers_total['beer_style']:
		if ('ipa' in style.lower()) or ('pale ale' in style.lower()):
			styles.append('IPA')
		elif ('wheat' in style.lower()):
			styles.append('Wheat')
		elif ('lager' in style.lower()):
			styles.append('Lager')
		elif ('porter' in style.lower()):
			styles.append('Porter')
		elif ('dark' in style.lower()) or ('stout' in style.lower()):
			styles.append('Stout')
		elif ('oktoberfest' in style.lower()):
			styles.append('Oktoberfest')
		elif ('pilsner' in style.lower()) or ('pilsener' in style.lower()):
			styles.append('Pilsner')
		elif ('sout' in style.lower()):
			styles.append('Sour')
		elif ('malt' in style.lower()):
			styles.append('Malt')
		elif ('wine' in style.lower()) or ('noir' in style.lower()):
			styles.append('Barleywine')
		elif ('scotch' in style.lower()):
			styles.append('Scotch')
		elif ('ginger' in style.lower()):
			styles.append('Ginger')
		elif ('ale' in style.lower()):
			styles.append('Ale')
		else:
			styles.append('Other')
	df_beers_total['beer_style'] = styles    
	return df_beers_total


@st.cache_data
def load_common_beers(df_beers_total, num_per_style = 15):
	'''
	Returns a dataframe to serve as a list of recommended beers
	df_beers_total: total beer rankings data
	num_per_style: can customize number of beers you want in the dropdown menu per beer style
	'''
	df_count = pd.pivot_table(df_beers_total, values = ['review_overall', 'beer_style', 'brewery_name'], index = ['beer_name'], aggfunc = {'review_overall': 'count', 'beer_style': 'max', 'brewery_name': 'max'})

	df_count = df_count.sort_values(by = 'review_overall', ascending = False)
	rate_these_beers_df = pd.DataFrame()
	for style in df_count['beer_style'].unique():
		if num_per_style > 0:
				df_style = df_count[df_count['beer_style'] == style].iloc[:num_per_style]
		else:
			df_style = df_count[df_count['beer_style'] == style]
		rate_these_beers_df = pd.concat([rate_these_beers_df, df_style])
	rate_these_beers_df.reset_index(level=0, inplace=True)
	rate_these_beers_df = rate_these_beers_df.sort_values(by = 'review_overall', ascending = False)
	rate_these_beers_df = rate_these_beers_df[['beer_name', 'beer_style', 'brewery_name']]
	return rate_these_beers_df



# Removed deprecated caching here because this function depends on a model object
# which is not suitable for st.cache_data/resource hashing.

def get_my_top_beers(name, beer_to_pred_rating, df_beers_total, df_my_beers, beer_style):
	'''
	Returns a dataframe of the top-recommended beers according to what SVD predicts
	name: User giving rankings
	beer_to_pred_rating: Dict mapping beer name to predicted rating for `name`
	df_beers_total: total beer rankings data
	df_my_beers: 
	beer_style: The style of beer to filter to
	'''
	my_beers = []
	for beer in df_beers_total[df_beers_total['beer_style'] == beer_style]['beer_name'].unique():
		if beer not in list(df_my_beers['beer_name'].unique()):
			est = beer_to_pred_rating.get(beer, 0.0)
			my_beers.append((beer, est))
	my_beers = sorted(my_beers, key = lambda x:x[1], reverse = True)
	return [x[0] for x in my_beers]



