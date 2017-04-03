from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, dok_matrix, find
from data_reader import output
import fileinput


class CreateData(object):
	"""Transform the raw data from set trec05 into instances.

	Raw data found in the ./data_reader/data/raw/trec05p-1/ base folder.
	Subsection of available data specified through subfolder name at the
	time of corpus creation. In future, can be extended to include more
	datasets, with variable ways of specifying n-fold data generation.

	"""

	def __init__(self, name):
		"""Create a new data generator.

		Extends the BaseModel class to use the functionality of
		a user-supplied sklearn classifier in conjunction with
		the adversarial library.

		Args:
		    name (str): name of the output files for test and train instances
		"""
		self.name = name            # type: str

		#: Path to raw data. @TODO: User-defined variable in future implementation
		self.base_path = './data_reader/data/raw/trec05p-1/'

		#: List of strings containing data for each email in dataset
		self.corpus = []

		#: True classification labels, as specified by raw data (ham vs. spam)
		self.labels = []

		#: Sklearn module used to fit and transform data and the resulting feature matrix
		self.tf = None              # type: TfidfVectorizer
		self.tfidf_matrix = None    # type: csr_matrix

		#: Number of instances within the current corpus
		self.num_instances = 0

	def create_corpus(self, folder):
		"""Generate list of files, one for each future instance.

        Args:
            folder (str): Path specifying %ham and %spam emails

        """
		# Reset stored values when making new corpus
		self.corpus = []
		self.labels = []
		self.num_instances = 0

		index_path = self.base_path + folder + '/index'
		files = self.find_files(index_path)

		for file in files:
			with open(file, 'r', encoding='utf-8', errors='replace', buffering=(2<<16) + 8) as email:
				email_string = email.read().replace('\n\t', ' ').replace('\n', ' ')
				self.corpus.append(email_string)

	def find_files(self, index_path):
		"""Generate list of file paths, one for each future instance.

        Args:
            index_path (str): Path containing file that specifies locations of raw files

        Returns:
            file paths (List(str))

        """
		files = []
		with open(index_path, 'r', buffering=(2<<16) + 8) as file_list:
			for line in file_list:
				category_path = line.replace('\n', '').split(' ../')
				if category_path[0] == 'spam':
					self.labels.append(1)
				else:
					self.labels.append(-1)
				files.append(self.base_path + category_path[1])
				self.num_instances += 1
		return files

	def tf_idf(self, strip_accents_=None, ngram_range_=(1, 1), max_df_=1.0, min_df_=1, max_features_=1000):
		"""Fit the term frequency model to the corpus

        Args:
            strip_accents_ (str): Accents to remove
            ngram_range_ (tuple): Range of phrase lengths to analyze
            max_df_ (float): Max degree of freedom for term frequencies
            min_df_(int): Min dgree of freedom for term frequencies
            max_features_: Maximim number of features to pull from the corpus

        """
		self.tf = TfidfVectorizer(analyzer='word', strip_accents=strip_accents_, ngram_range=ngram_range_,
		                          max_df=max_df_, min_df=min_df_, max_features=max_features_, binary=False,
		                          stop_words='english', use_idf=True, norm=None)
		self.tf = self.tf.fit(self.corpus)

	def create_instances(self, category):
		"""Use the fitted model to transform the corpus into feature vectors

        Args:
            category (str): Test or train

        """
		tfidf_matrix = self.tf.transform(self.corpus)
		tfidf_matrix.sort_indices()
		self.save_data(category, tfidf_matrix)

	def save_data(self, category, tfidf_matrix: csr_matrix):
		"""Output the generated feature vectors and known labels to file

        Args:
            category (str): Test or train
            tfidf_matrix (csr_matrix): num_instances X feature_count

        """
		instances = []

		indptr = tfidf_matrix.indptr
		indices = tfidf_matrix.indices

		for i in range(0, self.num_instances):
			instances.append([self.labels[i]]+indices[indptr[i]:indptr[i+1]].tolist())

		if category == 'all_categories':
			output.save_data('train', self.name, instances)
			output.save_data('test', self.name, instances)
		else:
			output.save_data(category, self.name, instances)
