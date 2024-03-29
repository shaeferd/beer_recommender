import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, SVD, accuracy
import sklearn.metrics as skmetric
from surprise.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from beer_reco_utils import load_beers, load_common_beers, get_my_top_beers


def main():
	
	# Display main beer image
	image = Image.open('beer_images/beer.jpg')
	st.image(image, use_column_width=True)

	# Load and cache beer rankings dataframe
	df_beers_total = load_beers()

	# Dropdown menu to select your favorite beer type
	"**\* = required**"
	beer_type = st.multiselect(
		"Select your favorite beer type (Optional)",
		list(set(df_beers_total['beer_style'])))
	if beer_type:
		rate_these_beers_df = load_common_beers(df_beers_total[df_beers_total['beer_style'].isin(beer_type)], num_per_style = 1000)
	else:
		rate_these_beers_df = load_common_beers(df_beers_total, num_per_style = 100)

	# Dropdown menu to select which beers to rate
	option = st.multiselect(
		"*Choose at least 5 beers to rate (the more the better)",
		list(rate_these_beers_df['beer_name']))

	# Make sure at least 5 beers are selected
	done_button = st.button('Submit Beers to Rate')
	ratings_button = False
	if done_button:
		if len(option) < 5:
			raise ValueError("Choose at least %s more beer(s)" % (5 - len(option)))
	beer_list = option
	rating_list = [''] * len(beer_list)
	beer_dict = dict(zip(beer_list, rating_list))

	# Returns a slider to rate each beer
	for k, v in beer_dict.items():
	    beer_dict[k] = st.slider(k, 1.0, 5.0, 3.0, .25)
	    st.write(beer_dict[k])

	# Turns your beer ratings into a dataframe, extracting the beer, rating, and style
	df_my_beers = pd.DataFrame(beer_dict.items(), columns = ['beer_name', 'review_taste'])
	my_styles = []
	for beer in df_my_beers['beer_name']:
		style = df_beers_total[df_beers_total['beer_name'] == beer]['beer_style'].values[0]
		my_styles.append(style)
	df_my_beers['beer_style'] = my_styles

	# Button to run recommendation algorithm
	model_button = False
	if len(df_my_beers) >= 5:
		model_button = st.button("Give me my beer recommendations")

	if model_button:
		"Searching for tasty beers..."

		# Predicts your favorite beer type based on your ratings and sorts in order of favorite to least favorite type
		beer_types = df_my_beers.groupby('beer_style')[['beer_style', 'review_taste']].mean()
		beer_types2 = df_my_beers.groupby('beer_style')[['beer_style', 'review_taste']].count()
		beer_types['review'] = beer_types2['review_taste']
		beer_types[['review_taste', 'review']] = MinMaxScaler().fit_transform(beer_types[['review_taste', 'review']])
		beer_types['overall'] = (beer_types['review_taste'] + beer_types2['review_taste']) / 2
		beer_types = beer_types.sort_values(by = 'overall', ascending = False)
		if beer_type:
			beer_type = beer_type
		else:
			beer_type = list(beer_types[beer_types['overall'] >= 1].index)
		df_beers_total = df_beers_total[df_beers_total['beer_style'].isin(beer_type)]
		df_my_beers['review_profilename'] = 'Me'
		df_my_beers = df_my_beers[['review_profilename', 'beer_name', 'review_taste']]

		# Combines user ranking data with total beer ranking data
		df_beers = pd.concat([df_beers_total[['review_profilename', 'beer_name', 'review_taste']], df_my_beers])
		df_beers = df_beers.groupby(['review_profilename', 'beer_name'])[['review_taste']].mean()
		df_beers.reset_index(level=0, inplace=True)
		df_beers.reset_index(level=0, inplace=True)
		df_beers = shuffle(df_beers)

		# Fits pre-tuned SVD recommender system to combined ratings data
		# In development phase, using 5-fold cross-validation, the average RMSE was 0.56 for the below parameters
		reader = Reader(rating_scale=(1, 5))
		data = Dataset.load_from_df(df_beers[['review_profilename', 'beer_name', 'review_taste']], reader)
		algo = SVD(n_epochs= 6, lr_all= 0.03, reg_all= 0.2)
		trainset = data.build_full_trainset()
		algo.fit(trainset)
		df_beers_total = df_beers_total[df_beers_total['beer_style'].isin(beer_type)]

		# Dictionary to map styles to images
		beer_type_map = {'IPA': 'beer_images/ipa.jpeg', 'Wheat': 'beer_images/wheat.jpg', 'Lager': 'beer_images/lager.jpg', 'Porter': 'beer_images/porter.jpg',
		'Stout': 'beer_images/stout.jpg', 'Oktoberfest': 'beer_images/oktoberfest.jpeg', 'Pilsner': 'beer_images/pilsner.jpg', 'Sour': 'beer_images/sour.jpg',
		'Malt': 'beer_images/malt.jpg', 'Barleywine': 'beer_images/barleywine.png', 'Scotch': 'beer_images/scotch.jpg', 'Ginger': 'beer_images/ginger.png', 'Ale': 'beer_images/ale.jpg', 'Other': 'beer_images/other.jpg'}
		

		# Loops through your favorite beer types and uses the SVD model to recommend 5 beers for each style
		for my_type in beer_type:
			num = 1
			recommended_beers = get_my_top_beers('Me', algo, df_beers_total, df_my_beers, my_type)
			"### Looks like you enjoy a nice %s beer! We highly recommend checking out these 5 beers below" % my_type
			beer_image = Image.open(beer_type_map[my_type])
			st.image(beer_image, width = 300)
			for beer in recommended_beers[:5]:
				"\#"+str(num) +' ' + str(beer)  
				num += 1
if __name__ == '__main__':
	"# Welcome to my beer recommendation app!"
	"### This app uses collaborative filtering to compare your beer preferences with others and recommend the tastiest of beers"
	main()


